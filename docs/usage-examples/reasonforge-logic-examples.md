# reasonforge-logic (13 tools)

**Symbolic AI, formal logic, and knowledge systems**

**Installation:** `pip install reasonforge-logic`

[View all packages](../../USAGE_EXAMPLES.md) | [Prev: reasonforge-physics](./reasonforge-physics-examples.md)

---

## Symbolic AI (6 tools)

### Example 98: Pattern to Equation

**User:** "Fit an equation to data points: [(0,1), (1,2), (2,5), (3,10)]"

**Claude with ReasonForge:**
```
Using pattern_to_equation tool from reasonforge-logic...

Result:
{
  "x_values": [0, 1, 2, 3],
  "y_values": [1, 2, 5, 10],
  "best_fit": "x**2 + 1",
  "r_squared": "0.9985",
  "candidates": [
    {"equation": "x**2 + 1", "score": "0.9985"},
    {"equation": "1.5*x**2 - 0.5*x + 1", "score": "0.9950"}
  ]
}
```

### Example 99: Symbolic Knowledge Extraction

**User:** "Extract logical rules from data: [(1,1,True), (1,0,False), (0,1,False), (0,0,False)]"

**Claude with ReasonForge:**
```
Using symbolic_knowledge_extraction tool from reasonforge-logic...

Result:
{
  "data_points": [
    {"x": 1, "y": 1, "output": true},
    {"x": 1, "y": 0, "output": false},
    {"x": 0, "y": 1, "output": false},
    {"x": 0, "y": 0, "output": false}
  ],
  "extracted_formula": "x & y",
  "interpretation": "Logical AND operation",
  "satisfiability": "satisfiable"
}
```

### Example 100: Symbolic Theorem Proving

**User:** "Prove that (A → B) ∧ A implies B"

**Claude with ReasonForge:**
```
Using symbolic_theorem_proving tool from reasonforge-logic...

Result:
{
  "premises": ["Implies(A, B)", "A"],
  "goal": "B",
  "proof_found": true,
  "method": "modus_ponens",
  "proof_steps": [
    "Given: A → B",
    "Given: A",
    "By Modus Ponens: B"
  ],
  "confidence": "100%"
}
```

### Example 101: Feature Extraction

**User:** "Extract a rule for even numbers from examples"

**Claude with ReasonForge:**
```
Using feature_extraction tool from reasonforge-logic...

Result:
{
  "positive_examples": [2, 4, 6, 8],
  "negative_examples": [1, 3, 5, 7],
  "learned_rule": "x mod 2 = 0",
  "confidence": "1.0",
  "coverage": "100% of positive examples, 0% of negative examples"
}
```

### Example 102: Structure Mapping

**User:** "Find analogy: addition is to subtraction as multiplication is to ?"

**Claude with ReasonForge:**
```
Using structure_mapping tool from reasonforge-logic...

Result:
{
  "source_domain": {
    "operation1": "addition",
    "inverse1": "subtraction"
  },
  "target_domain": {
    "operation2": "multiplication",
    "inverse2": "?"
  },
  "mapping": {"inverse2": "division"},
  "analogy_strength": "0.95",
  "explanation": "Multiplication and division are inverse operations, analogous to addition and subtraction"
}
```

### Example 103: Automated Conjecture

**User:** "Generate conjectures about prime numbers"

**Claude with ReasonForge:**
```
Using automated_conjecture tool from reasonforge-logic...

Result:
{
  "domain": "number_theory",
  "context_objects": ["prime"],
  "conjectures": [
    {
      "statement": "For all primes p > 2, p mod 2 = 1",
      "verified": true,
      "counterexamples": []
    },
    {
      "statement": "The sum of two primes is always even (except 2+odd)",
      "verified": true,
      "counterexamples": []
    },
    {
      "statement": "All primes > 3 can be written as 6k±1",
      "verified": true,
      "counterexamples": []
    }
  ]
}
```

## Formal Logic (4 tools)

### Example 104: First Order Logic

**User:** "Parse the FOL formula: ∀x (P(x) → ∃y Q(x,y))"

**Claude with ReasonForge:**
```
Using first_order_logic tool from reasonforge-logic...

Result:
{
  "operation": "parse",
  "formula": "ForAll(x, Implies(P(x), Exists(y, Q(x, y))))",
  "parsed": true,
  "free_variables": [],
  "bound_variables": ["x", "y"],
  "latex": "\\forall x (P(x) \\rightarrow \\exists y Q(x,y))"
}
```

### Example 105: Propositional Logic Advanced

**User:** "Convert (A ∧ B) ∨ C to CNF"

**Claude with ReasonForge:**
```
Using propositional_logic_advanced tool from reasonforge-logic...

Result:
{
  "operation": "cnf",
  "formula": "(A ∧ B) ∨ C",
  "cnf": "(A | C) & (B | C)",
  "latex": "\\left(A \\vee C\\right) \\wedge \\left(B \\vee C\\right)",
  "description": "Conjunction of disjunctions (AND of ORs)",
  "result": "(A | C) & (B | C)"
}
```

### Example 106: Modal Logic

**User:** "Validate the K axiom: □(p → q) → (□p → □q)"

**Claude with ReasonForge:**
```
Using modal_logic tool from reasonforge-logic...

Result:
{
  "logic_type": "K",
  "formula": "Implies(Box(Implies(p, q)), Implies(Box(p), Box(q)))",
  "operation": "validate",
  "valid": true,
  "interpretation": "Distribution axiom of modal logic K"
}
```

### Example 107: Fuzzy Logic

**User:** "Compute fuzzy intersection of sets A={x:0.7, y:0.3} and B={x:0.5, y:0.9}"

**Claude with ReasonForge:**
```
Using fuzzy_logic tool from reasonforge-logic...

Result:
{
  "operation": "intersection",
  "fuzzy_set_a": {"x": 0.7, "y": 0.3},
  "fuzzy_set_b": {"x": 0.5, "y": 0.9},
  "result": "μA∩B(x) = min(μA(x), μB(x))",
  "computed_intersection": {"x": 0.5, "y": 0.3}
}
```

## Knowledge Systems (2 tools)

### Example 108: Knowledge Graph Reasoning

**User:** "Find transitive closure of edges: [(A,B), (B,C)]"

**Claude with ReasonForge:**
```
Using knowledge_graph_reasoning tool from reasonforge-logic...

Result:
{
  "operation": "transitive_closure",
  "edges": [["A", "B"], ["B", "C"]],
  "reasoning_results": ["Transitive closure: If (A→B) and (B→C) then (A→C)"],
  "transitive_edges": [["A", "B"], ["B", "C"], ["A", "C"]],
  "result": [["A", "B"], ["B", "C"], ["A", "C"]]
}
```

### Example 109: Constraint Satisfaction

**User:** "Solve CSP: variables={A,B,C}, domains={A:[1,2], B:[2,3], C:[1,3]}, constraints=[A<B, B<C]"

**Claude with ReasonForge:**
```
Using constraint_satisfaction tool from reasonforge-logic...

Result:
{
  "variables": ["A", "B", "C"],
  "domains": {"A": [1, 2], "B": [2, 3], "C": [1, 3]},
  "constraints": ["A < B", "B < C"],
  "solution": {"A": 1, "B": 2, "C": 3},
  "result": {"A": 1, "B": 2, "C": 3},
  "satisfiable": true,
  "method": "Constraint propagation and backtracking"
}
```

## Proof Generation (1 tool)

### Example 110: Generate Proof

**User:** "Prove that the sum of two even numbers is even"

**Claude with ReasonForge:**
```
Using generate_proof tool from reasonforge-logic...

Result:
{
  "theorem": "The sum of two even numbers is even",
  "axioms": ["An integer n is even iff n = 2k for some integer k"],
  "proof_steps": [
    "Let a and b be even integers.",
    "By definition of even, there exist integers j and k such that a = 2j and b = 2k.",
    "Consider their sum: a + b = 2j + 2k = 2(j + k).",
    "Since j + k is an integer (closure of addition), let m = j + k.",
    "Thus a + b = 2m, where m is an integer.",
    "By definition, a + b is even. ∎"
  ],
  "proven": true,
  "proof_type": "direct",
  "proof_method": "structured_proof"
}
```

---

[View all packages](../../USAGE_EXAMPLES.md) | [Prev: reasonforge-physics](./reasonforge-physics-examples.md)
