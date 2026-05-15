"""LLM prompts used by the atomization stage."""

# Stage 1: Blog generation from PDF
BLOG_GENERATION_PROMPT = """<task_description>
Convert this research paper into an intuitive blog post that explores the main ideas of the paper.
</task_description>

<requirements>
- Length: Approximately 2 pages
- Focus: Methodology only (how it works)
- Exclude: Experiments, results, and empirical evaluations
- Style: Clear and accessible explanations suitable for a technical blog audience
</requirements>

<output_format>
Write the blog post in markdown format with appropriate headings and structure.
</output_format>"""

# Stage 2: Idea extraction from blog
IDEA_EXTRACTION_PROMPT = """<task_description>
Extract key ideas from the following blog post describing a research paper.
These ideas will be used to find conceptual connections across different research papers. Each idea must work as a standalone "atom of knowledge" that could meaningfully cluster with ideas from completely different papers.
</task_description>

<blog_post>
{blog_content}
</blog_post>

<idea_requirements>
Each idea must be:
1. Self-standing: Understandable without context from other extracted ideas
2. No cross-references: Must not assume knowledge defined in other ideas from the same extraction
3. Method names as anchors, not crutches: Method names can be included to enable clustering, but the idea must remain understandable even if the reader has never heard that name
4. Specific enough to be useful: Concrete techniques and mechanisms, but general enough to cluster with ideas from other papers
5. Complete: Include both the mechanism (what the technique does) and sufficient context that a researcher from a different field could understand why it matters

Each idea should pass the "foreign reader test": a researcher from a different subfield should be able to understand it and potentially connect it to their own work without having read the source paper.
</idea_requirements>

<examples>
GOOD IDEAS:

"Associative memory can be modeled as a sparse recovery problem where a network learns linear constraints that all valid memories must satisfy, then uses those constraints to identify and remove noise from corrupted input signals."
Reason: Explains the reframing, the mechanism, and the purpose. No dependencies on other ideas.

"The Method of Auxiliary Coordinates (MAC) decouples embedding objectives from mapping functions by introducing temporary placeholder coordinates that are optimized separately, allowing any embedding algorithm (like t-SNE) to be paired with any function approximator (like a neural network or decision tree)."
Reason: Names the method for clustering, but fully explains the mechanism so it also clusters with "decoupling" or "auxiliary variable" concepts from other fields.

"Greedy algorithms for monotone submodular maximization guarantee that the solution is at least 63% as good as the theoretical optimum, because submodularity's diminishing returns property bounds how much value can be lost by making locally optimal choices."
Reason: States the result and explains why it holds. Would cluster with approximation guarantees across optimization.


BAD IDEAS:

"MAC provides universal compatibility by allowing researchers to mix and match different embedding objectives."
Reason: Method-name as crutch. A reader who doesn't know MAC learns nothing actionable.

"The 'Z' step optimizes coordinates using Barnes-Hut approximation."
Reason: Assumes you know there's a Z-step. This is a procedural fragment, not a standalone insight.

"This significantly improves performance over the baseline."
Reason: Dangling "this," no specifics about what technique or why it helps.

"The algorithm prunes regions whose bounds fall below the best solution."
Reason: "Regions" and "bounds" are unexplained. What kind of regions? Bounds on what?
</examples>

<extraction_guidance>
- Prefer conceptual insights over procedural details
- If an idea requires a definition to make sense, include that definition inline
- If two ideas depend on each other, merge them into one or drop the weaker one
- Extract only ideas that genuinely meet the quality criteria—fewer strong ideas are better than many weak ones
</extraction_guidance>

<output_format>
Return ONLY a valid JSON object in this exact format (no markdown code blocks, no additional text):
{{
  "ideas": [
    "First self-contained idea here",
    "Second self-contained idea here"
  ]
}}

Ensure all text is properly escaped for JSON.
</output_format>"""


IDEA_RATING_AND_REFINEMENT_PROMPT = """<task_description>
You are evaluating and refining extracted "atoms of knowledge" from research papers. For each idea, you will:
1. Analyze its quality as a standalone, transferable insight
2. Rate it as "strong", "medium", or "weak"
3. For medium or weak ideas, provide a revised version that addresses the issues

The goal is to maximize the number of usable, high-quality atoms—not to filter aggressively.
</task_description>

<ideas_to_rate>
{ideas_json}
</ideas_to_rate>

<rating_criteria>
**strong**: 
- Fully self-contained—understandable without any external context
- Passes the "foreign reader test": a researcher from a different field could understand it
- Includes both mechanism (what) and sufficient context (why it matters)
- No dangling references ("this approach", "the method" without antecedent)
- Method names, if present, are explained rather than assumed
- Likely to cluster meaningfully with related ideas from other domains

**medium**:
- Mostly self-contained but has minor issues
- Perhaps slightly too specific to the source paper
- Or one technical term could use more context
- Or the significance isn't fully clear
- Fixable with modest revision

**weak**:
- Has significant issues that undermine standalone use
- Dangling references or unexplained terms
- Depends on knowledge from other ideas or the source paper
- Method name used as a crutch without explanation
- Too procedural (describes steps rather than insight)
- May still contain a valuable insight worth rescuing through substantial revision
</rating_criteria>

<refinement_guidance>
For medium and weak ideas, provide a revised version that:
- Adds missing context or definitions inline
- Replaces dangling references with explicit descriptions
- Expands method names into explanations
- Converts procedural descriptions into transferable insights
- Clarifies the mechanism and why it matters

If an idea is fundamentally unrescuable (e.g., too vague to interpret, or purely procedural with no underlying insight), set revised_idea to null.

Do not over-expand: the revised idea should still be concise (1-3 sentences). Add only what's necessary for self-containment.
</refinement_guidance>

<rating_guidance>
Be critical in your ratings but constructive in your revisions.

For each idea, ask:
1. If I showed this to a researcher who hasn't read the paper, would they understand it?
2. Does every noun and technical term have enough context?
3. Is there a "this" or "the method" without a clear antecedent?
4. Would this cluster with ideas from other papers, or only with ideas from the same paper?
5. Is a method name doing the explanatory work, or is the mechanism actually explained?

IMPORTANT: Write your rationale FIRST, then decide on the rating, then write the revision if needed.
</rating_guidance>

<examples>
Example idea: "Associative memory can be modeled as a sparse recovery problem where a network learns linear constraints that all valid memories must satisfy, then uses those constraints to identify and remove noise from corrupted input signals."
Rationale: This idea explains a complete reframing (memory as sparse recovery), includes the mechanism (learn constraints, use them to remove noise), and states the purpose (retrieve clean data from corrupted input). No external knowledge required. A researcher in signal processing or optimization could immediately see connections to their own work.
Rating: strong
Revised idea: null

Example idea: "The 'Z' step optimizes coordinates using Barnes-Hut approximation."
Rationale: This assumes the reader knows there is a "Z step" as part of some larger algorithm. Without that context, "Z step" is meaningless. The idea describes a procedural detail rather than a transferable insight. However, the underlying insight about using N-body approximations for embedding optimization is valuable.
Rating: weak
Revised idea: "N-body approximation methods like Barnes-Hut can accelerate embedding optimization by efficiently computing repulsive forces between all pairs of points in O(N log N) time instead of O(N²), enabling non-linear dimensionality reduction techniques to scale to millions of data points."

Example idea: "Projecting onto the Top-k Simplex can be done in O(m log m) time where m is the number of classes."
Rationale: The complexity result is clearly stated and the variable is defined. However, "Top-k Simplex" is a specific geometric object that isn't explained—a reader unfamiliar with this paper wouldn't know what shape or constraint this refers to.
Rating: medium
Revised idea: "Projecting onto a simplex where no single coordinate can exceed 1/k of the total (forcing probability mass to spread across at least k classes) can be computed in O(m log m) time through sorting and threshold search, enabling efficient optimization of top-k classification objectives."

Example idea: "This improves performance significantly."
Rationale: Completely unrescuable. No indication of what technique is being discussed, what kind of performance, or why improvement occurs. There is no underlying insight to extract.
Rating: weak
Revised idea: null
</examples>

<output_format>
Return ONLY a valid JSON object in this exact format:
{{
  "ratings": [
    {{
      "idea": "The exact original idea text",
      "rationale": "Your analysis of the idea's strengths and weaknesses",
      "quality": "strong|medium|weak",
      "revised_idea": "Improved version of the idea, or null if strong or unrescuable"
    }}
  ]
}}

Maintain the same order as the input ideas.
</output_format>"""


# Stage 4: Cluster naming
CLUSTER_NAMING_PROMPT = """<task_description>
You are analyzing a cluster of "atoms of knowledge" extracted from various research papers. 
Your goal is to synthesize these related insights into a "Super Atom"—a single, high-level conceptual insight that captures the shared mechanism and value of the entire cluster.
</task_description>

<cluster_atoms>
{atoms_text}
</cluster_atoms>

<super_atom_requirements>
The Super Atom must:
1. **Be an "Atom" itself**: It must pass the "foreign reader test." A researcher from a different field should understand the insight without seeing the individual atoms.
2. **Synthesize, don't summarize**: Do not just list what the atoms say. Identify the underlying principle, methodology, or conceptual breakthrough they all share.
3. **Mechanism + Purpose**: Explicitly state *how* the shared concept works and *why* it is significant across the contexts provided.
4. **Self-standing**: Use no "this," "these," or "the aforementioned." It must be a complete, independent unit of knowledge.
5. **Level of Abstraction**: It should be specific enough to be technical and "crunchy," but general enough to explain why these disparate ideas belong together.
</super_atom_requirements>

<examples>
INPUT ATOMS:
- "Residual connections in CNNs allow gradients to bypass layers, preventing the vanishing gradient problem in very deep networks."
- "Highway Networks use gating mechanisms to control information flow across layers, enabling the training of networks with hundreds of layers."
- "Dense blocks connect every layer to every subsequent layer to ensure maximum information flow and feature reuse."

BAD SUPER ATOM: "Various methods for connecting layers in deep neural networks to improve training." (Too vague, no mechanism).

GOOD SUPER ATOM (The Synthesis): "Architectural shortcuts—such as residual, gated, or dense connections—mitigate the vanishing gradient problem by creating direct paths for information and gradient flow, allowing deep hierarchical models to maintain signal integrity during training."
</examples>

<output_format>
Return ONLY a JSON object in this format:
{{
  "rationale": "Briefly explain the common thread you identified",
  "super_atom": "The synthesized, high-level atom of knowledge",
  "coherence_score": "low|medium|high"
}}
</output_format>"""
