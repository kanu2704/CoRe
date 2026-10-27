"""Soft success labels for P-SVM (mode rrm_psvm).

At the end of every episode the VLM sees the task text, CoRe's task-initialisation and
task-completion images and frames from the episode, and grades how far the attempt got.
q in [0, 1] is the episode's success weight in the SVM: positives are sampled in proportion
to q, negatives to 1 - q. Near-misses get partial credit, so the SVM can switch on before
the first real success, which is where the oracle-label SVM (rrm_svm) gets stuck.

Grading styles:
  stages  (default) three frames of the attempt and a checklist of the task's stages; the
          VLM names the furthest stage reached and q = stage / 4
  success the same three frames; the VLM answers whether the task was completed, q = 1 or 0
  scale   the final frame only, graded 0-10 (q = grade / 10). On Peg Insert Qwen-4B gave 0
          to every attempt that had not finished the task, so it never gave partial credit.
"""
import os
import re
import time

import cv2

from FRM.code_agent import DailyQuotaError, is_daily_quota, is_rate_limit, make_client, pace
from RRM.preference_label import np_to_base64_image
from RRM.prompt import env_goal_prompts

SCALE_PROMPT = ("The task is {task}. The first image shows the task not started at all, and the second image "
                "shows the task completely done. The third image is the final frame of one attempt. "
                "Looking only at the objects involved in the task, how far did this attempt get? "
                "Answer with a single integer from 0 (same as not started) to 10 (completely done, like the second image).")

STAGES_PROMPT = ("The task is {task}. The first image shows the task not started at all, and the second image "
                 "shows the task completely done. The next {n} images are frames from one attempt, in time order. "
                 "The robot arm is invisible in all images, so judge only by where the objects are. "
                 "Which is the furthest stage the attempt reached in any of these frames?\n{stages}\n"
                 "Answer with a single integer from 0 to 4.")

SUCCESS_PROMPT = ("The task is {task}. The first image shows the task not started at all, and the second image "
                  "shows the task completely done. The next {n} images are frames from one attempt, in time order; "
                  "the last one is the end of the attempt. The robot arm is invisible in all images, so judge only "
                  "by where the objects are. Was the task completed, as in the second image? Answer with one word: yes or no.")


def parse_yes_no(text):
    m = re.search(r'\b(yes|no)\b', (text or '').lower())
    return None if m is None else (1 if m.group(1) == 'yes' else 0)


# what each stage looks like, per task (stage 4 = done). CoRe renders the robot arm transparent,
# so the stages describe only the objects.
STAGES = {
    'metaworld_peg-insert-side-v2': [
        "the green peg is still where it started",
        "the green peg has moved a little from where it started",
        "the green peg is about halfway from its start to the red block",
        "the green peg is right next to the hole in the red block",
        "the green peg is inside the hole in the red block, like in the second image"],
    'metaworld_hammer-v2': [
        "the red hammer is still where it started",
        "the hammer has moved a little from where it started",
        "the hammer is about halfway from its start to the grey nail",
        "the hammer's head is right next to the grey nail",
        "the grey nail is hammered in, like in the second image"],
    'metaworld_button-press-topdown-v2': [
        "the red button is fully up, as at the start",
        "the red button is pressed down a little",
        "the red button is pressed about halfway down",
        "the red button is almost flat",
        "the red button is pressed down flat, like in the second image"],
}


def parse_grade(text, top=10):
    """'7', 'Grade: 7', '7/10', '7 out of 10', 'On a scale of 0 to 10, I'd give 7' -> 7; None if no grade."""
    text = text or ''
    m = re.search(r'(?<!\d)(10|[0-9])\s*(?:/|out of)\s*(?:10|4)(?!\d)', text)
    if m:
        g = int(m.group(1))
    else:
        nums = re.findall(r'(?<!\d)(10|[0-9])(?!\d)', text)
        g = int(nums[-1]) if nums else None  # the last number: a scale ('0 to 10') comes before the answer
    return g if g is not None and g <= top else None


class SuccessJudge:
    def __init__(self, env_name, project_dir, style='stages', model=None):
        if style == 'stages' and env_name not in STAGES:
            raise ValueError(f"no stage checklist for {env_name}; add one to psvm/judge.py or use --judge_style scale")
        self.client = make_client('vlm')
        self.model = model or os.environ.get('VLM_MODEL', 'gemini-2.5-flash-lite')
        self.style = style
        self.task = env_goal_prompts[env_name]
        self.stages = '\n'.join(f"{i}: {s}" for i, s in enumerate(STAGES.get(env_name, [])))
        image = lambda kind: np_to_base64_image(cv2.resize(cv2.cvtColor(
            cv2.imread(os.path.join(project_dir, 'RRM', 'goal_image', f'{env_name}-{kind}.jpg')),
            cv2.COLOR_BGR2RGB), (300, 300)))
        self.init_image, self.done_image = image('init'), image('done')
        self.calls = self.failures = 0
        self.last_answer = None

    def messages(self, frames):
        url = lambda b64: {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}}
        if self.style == 'scale':
            text, frames = SCALE_PROMPT.format(task=self.task), frames[-1:]
        elif self.style == 'success':
            text = SUCCESS_PROMPT.format(task=self.task, n=len(frames))
        else:
            text = STAGES_PROMPT.format(task=self.task, n=len(frames), stages=self.stages)
        return [{'role': 'user', 'content': [{'type': 'text', 'text': text}, url(self.init_image), url(self.done_image)]
                 + [url(np_to_base64_image(f)) for f in frames]}]

    def grade(self, frames):
        """q in [0, 1] for an attempt shown by `frames` (RGB uint8, time order, the last one final),
        or None if the VLM gave no usable answer."""
        top = {'scale': 10, 'stages': 4, 'success': 1}[self.style]
        messages = self.messages(frames)
        self.calls += 1
        for attempt in range(6):
            try:
                pace('vlm')
                r = self.client.chat.completions.create(model=self.model, messages=messages,
                                                        temperature=0, max_tokens=16)
                text = r.choices[0].message.content
                self.last_answer = text
                g = parse_yes_no(text) if self.style == 'success' else parse_grade(text, top)
                if g is not None:
                    return g / top
            except Exception as e:
                if is_daily_quota(e):
                    raise DailyQuotaError(f"{self.model}: daily quota used up. {str(e)[:300]}") from e
                time.sleep(min(60, 5 * 2 ** attempt) if is_rate_limit(e) else 2)
        self.failures += 1
        return None
