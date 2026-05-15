"""Prompts used by the generation and reconstruction stage."""

RECONSTRUCTION_PROMPT = """<task_description>
You are reconstructing a research methodology from high-level conceptual insights called "super-atoms."

Each super-atom is a transferable idea extracted from research papers. Your task is to synthesize these atoms into a coherent methodology that could have produced all of them together.

Key insight: You do not know which specific paper these atoms came from. Instead, infer what methodology would naturally incorporate all the given mechanisms. Think: "What research approach would a team design if they wanted to use all these techniques/ideas together?"
</task_description>

<super_atoms>
{super_atoms_text}
</super_atoms>

<requirements>
Synthesis goal:
- Infer a coherent methodology that explains all the super-atoms.
- The methodology should feel like a unified research contribution, not a list of techniques.
- Connect the atoms: how do they work together, and what problem do they collectively solve?

Content focus:
- Describe the mechanism: what does the approach do step by step?
- Explain why each component matters and how the components interact.
- Be concrete enough that a reader could implement the core technique.

Style:
- Technical blog suitable for ML practitioners.
- About two pages, similar to the original research blogs.
- Clear structure: problem -> approach -> key mechanisms -> why it works.

Constraints:
- Use only information from the super-atoms above.
- Do not invent specific method names, datasets, or numbers.
- Do not drop any atoms; find how they all fit together.
</requirements>

<output_format>
Return the blog post as plain markdown text. Do not wrap it in JSON or code blocks.
</output_format>"""


LLM_SELECTION_PROMPT = """<task_description>
You are selecting super-atoms (high-level research concepts) to form a coherent and novel research methodology.

Given a list of available super-atoms, select {n_select} that work together as components of a unified research approach.
</task_description>

<available_super_atoms>
{super_atoms_list}
</available_super_atoms>

<selection_criteria>
- Coherence: selected atoms should logically connect into a plausible methodology.
- Novelty: prefer combinations that are likely to create a novel methodology, one that no scientist would have thought of before.
- Completeness: together, the selected atoms should suggest a complete research approach.
</selection_criteria>

<output_format>
Return only a valid JSON object:
{{
  "reasoning": "Brief explanation of why these super-atoms work together",
  "selected_ids": [id1, id2, id3, ...]
}}

The selected_ids must be exact cluster IDs from the available list.
</output_format>"""
