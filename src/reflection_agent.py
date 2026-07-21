class ReflectionAgent:
    def __init__(self, llm):
        self.llm = llm

    def decide(
        self,
        query,
        context,
        draft_answer,
        current_entity=None
    ):
        query_lower = query.lower()

        pronouns = [
            " he ",
            " she ",
            " him ",
            " her ",
            " his ",
            " its ",
            " it "
        ]

        has_pronoun = any(
            p in f" {query_lower} "
            for p in pronouns
        )

        if has_pronoun and not current_entity:
            return "CLARIFY"

        prompt = f"""
You are a retrieval QA critic.

Your task is to evaluate whether the answer is good.

Question:
{query}

Retrieved Context:
{context}

Draft Answer:
{draft_answer}

Choose EXACTLY ONE action:

ACCEPT
REGENERATE
RETRIEVE_AGAIN
REFUSE

Decision rules:

- ACCEPT:
  if answer is grounded, relevant, and answers the question

- REGENERATE:
  if context is relevant but answer quality is weak,
  contradictory, vague, or poorly phrased

- RETRIEVE_AGAIN:
  if context seems weak, irrelevant, or incomplete

- REFUSE:
  if document clearly does not contain the answer

Return ONLY the action word.
"""

        try:
            response = self.llm.invoke(prompt)
            decision = response.content.strip().upper()

            valid = {
                "ACCEPT",
                "REGENERATE",
                "RETRIEVE_AGAIN",
                "REFUSE"
            }

            if decision not in valid:
                return "ACCEPT"

            return decision
        except Exception as e:
            print(f"[reflection] LLM evaluation error fallback to ACCEPT: {e}")
            return "ACCEPT"