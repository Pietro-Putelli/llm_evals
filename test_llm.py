from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval import assert_test
from local_model import LocalModel
import json

data = []
with open("dataset.json", "r") as file:
    data = json.load(file)

def test_case():
    local_model = LocalModel()

    actual_output = local_model.generate(
        f"""
            Your task: Refine raw voice dictation into clear, professional, grammatically flawless text. Preserve the original meaning, numbers, and language.

            Context: The input is the result of voice dictation, so it may contain:
            - Incorrect grammar or punctuation
            - Misheard or misinterpreted words (e.g., “access” instead of “assess”)
            - Missing structure like sentence breaks or list formatting

            Key Transformations:
            * Correct grammar, spelling, and punctuation.
            * The input is a result of voice dictation, so it may contain errors or words misinterpreted by the speech recognition system.
            * Replace the misinterpreted words with the correct ones related to the context.
            * Improve sentence structure and add paragraph breaks.
            * Format dictated lists (bullets/numbers) even if they appear as inline lists with multiple items (e.g., separated by commas or 'and'), if the context suggests a checklist, steps, or distinct items.
            * Ensure consistent capitalization.
            * Add descriptive headings when indicated.
            * Keep it concise.

            Deliver only the refined text, no commentary. Do not add information, distort meaning, over-correct, or change the original language.
            Don't include "Here's the refined text:" or similar phrases. Just provide the refined text directly.

            Text to refine: {data[12]["input"]}
        """
    )

    print(actual_output)

    correctness_metric = GEval(
        name="Correctness",
        criteria="""
        A high-quality response must:
        - Refine raw dictation into clear, professional, grammatically correct text.
        - Preserve meaning, numbers, names, and original language.
        - Correct grammar, spelling, punctuation, and obvious speech-to-text errors.
        - Improve sentence structure, add paragraph breaks, and format lists when context suggests.
        - Output only the refined text—no commentary or meta text.

        Fail if:
        - Meaning, numbers, or content are changed.
        - Errors remain uncorrected.
        - Extra commentary or filler is added.
        - Language, tone, or terminology are altered inappropriately.
        - Structure is missing or headings are added without clear indication.
        """,
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5,
    )

    test_case = LLMTestCase(
        input=data[12]["input"],
        actual_output=actual_output,
        expected_output=data[12]["expected"],
    )

    assert_test(test_case, [correctness_metric])
