from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval import assert_test
from local_model import LocalModel

TEXTS = [
    {
        "input": "slab thickness should be 4 in no wait 4.5 inch make sure use rebar number three or maybe number four spacing every 16 center",
        "expected": """
            **Concrete Slab Thickness Specifications**

            The recommended slab thickness is 4 inches. However, a maximum of 4.5 inches should be used.

            When incorporating rebar, utilize either rebar number three or rebar number four. Maintain a consistent spacing of 16 inches between the rebar.
        """
    },
    {
        "input": "order list wood screws nails drywall then for plumbing copper pipe pvc joints faucets sinks toilets and for electrical wire breakers switches",
        "expected": """
            **Materials List**
            The following materials are required for the project:

            *   Wood screws
            *   Nails
            *   Drywall
            *   Plumbing:
                *   Copper pipe
                *   PVC joints
                *   Faucets
                *   Sinks
                *   Toilets
            *   Electrical:
                *   Wire
                *   Breakers
                *   Switches
        """
    },
    {
        "input": "client said she want kitchen more open concept move sink from window wall to island add dishwasher space upgrade counter top to quartz add backsplash tile she mentioned maybe subway tile or mosaic she not decide also wants list of options appliances fridge stove oven microwave dishwasher and want cost comparison between stainless steel and built-in panel ready",
        "expected": """
            Client requests modifications to the kitchen design, aiming for a more open concept. Specifically, she wants to:

            *   Move the sink from the window wall to the island.
            *   Add dishwasher space.
            *   Upgrade the countertop to quartz.
            *   Add a backsplash tile, potentially subway tile or mosaic. She has not yet decided.

            The client also requires a list of appliance options, including:

            *   Refrigerator
            *   Stove
            *   Oven
            *   Microwave
            *   Dishwasher

            A cost comparison between stainless steel and built-in panel-ready appliances is also requested.
        """
    }
]

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

            Text to refine: {TEXTS[2]["input"]}
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
        input=TEXTS[2]["input"],
        actual_output=actual_output,
        expected_output=TEXTS[2]["expected"],
    )

    assert_test(test_case, [correctness_metric])
