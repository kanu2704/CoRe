"""Soft success labels for P-SVM (mode rrm_psvm).

At the end of every episode the VLM sees the task text, CoRe's task-initialisation and
task-completion images, and the episode's final frame, and grades how far the attempt got
from 0 (not started) to 10 (completed). q = grade / 10 is the episode's success weight in
the SVM: positives are sampled in proportion to q, negatives to 1 - q. Near-misses get
partial credit, so the SVM can switch on before the first real success, which is where the
oracle-label SVM (rrm_svm) gets stuck on hard tasks.
"""
import os
import re
import time

import cv2

from FRM.code_agent import DailyQuotaError, is_daily_quota, is_rate_limit, make_client, pace
from RRM.preference_label import np_to_base64_image
from RRM.prompt import env_goal_prompts

PROMPT = ("The task is {task}. The first image shows the task not started at all, and the second image "
          "shows the task completely done. The third image is the final frame of one attempt. "
          "Looking only at the objects involved in the task, how far did this attempt get? "
          "Answer with a single integer from 0 (same as not started) to 10 (completely done, like the second image).")


def parse_grade(text):
    """'7', 'Grade: 7', '7/10', '7 out of 10', 'On a scale of 0 to 10, I'd give 7' -> 7; None if no grade."""
    text = text or ''
    m = re.search(r'(?<!\d)(10|[0-9])\s*(?:/|out of)\s*10(?!\d)', text)
    if m:
        return int(m.group(1))
    nums = re.findall(r'(?<!\d)(10|[0-9])(?!\d)', text)
    return int(nums[-1]) if nums else None  # the last number: a scale ('0 to 10') comes before the answer


class SuccessJudge:
    def __init__(self, env_name, project_dir, model=None):
        self.client = make_client('vlm')
        self.model = model or os.environ.get('VLM_MODEL', 'gemini-2.5-flash-lite')
        self.task = env_goal_prompts[env_name]
        image = lambda kind: np_to_base64_image(cv2.cvtColor(
            cv2.imread(os.path.join(project_dir, 'RRM', 'goal_image', f'{env_name}-{kind}.jpg')), cv2.COLOR_BGR2RGB))
        self.init_image, self.done_image = image('init'), image('done')
        self.calls = self.failures = 0

    def grade(self, frame):
        """q in [0, 1] for an episode whose final frame is `frame` (RGB uint8), or None if the VLM
        gave no usable answer."""
        url = lambda b64: {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}}
        messages = [{'role': 'user', 'content': [
            {'type': 'text', 'text': PROMPT.format(task=self.task)},
            url(self.init_image), url(self.done_image), url(np_to_base64_image(frame))]}]
        self.calls += 1
        for attempt in range(6):
            try:
                pace('vlm')
                r = self.client.chat.completions.create(model=self.model, messages=messages,
                                                        temperature=0, max_tokens=16)
                g = parse_grade(r.choices[0].message.content)
                if g is not None:
                    return g / 10.0
            except Exception as e:
                if is_daily_quota(e):
                    raise DailyQuotaError(f"{self.model}: daily quota used up. {str(e)[:300]}") from e
                time.sleep(min(60, 5 * 2 ** attempt) if is_rate_limit(e) else 2)
        self.failures += 1
        return None
