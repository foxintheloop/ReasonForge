#!/usr/bin/env python3
"""
ReasonForge Hard Test Suite
===========================

Test cases specifically designed to exceed Claude's native reasoning capabilities.
These exploit cognitive limitations:
- Working memory limits (7±2 items)
- Combinatorial explosion
- Subtle constraint interactions
- Counter-intuitive results

Categories:
1. SAT - Satisfiability problems
2. KG - Knowledge Graph transitive closure
3. CSP - Constraint Satisfaction Problems
"""

# =============================================================================
# CATEGORY 1: SAT PROBLEMS
# =============================================================================

SAT_HARD_TESTS = [
    # -------------------------------------------------------------------------
    # Test 1: Deceptive SAT - Looks satisfiable but isn't
    # -------------------------------------------------------------------------
    {
        "id": "sat_deceptive_1",
        "name": "Deceptive UNSAT",
        "difficulty": "hard",
        "description": """
        This formula has many near-solutions. Each clause individually seems 
        easy to satisfy, but the combination is impossible. Native reasoning
        tends to find partial solutions and assume success.
        """,
        "formula": "(A | B) & (A | ~B) & (~A | B) & (~A | ~B)",
        "expected": {
            "satisfiable": False,
            "explanation": "Requires A and ~A simultaneously"
        },
        "why_hard": "Each clause pair seems independent, masks the contradiction"
    },
    
    # -------------------------------------------------------------------------
    # Test 2: Hidden dependency chain
    # -------------------------------------------------------------------------
    {
        "id": "sat_chain_5",
        "name": "5-Variable Implication Chain",
        "difficulty": "hard", 
        "description": """
        A→B→C→D→E chain with A forced true and E forced false.
        Native reasoning must track all implications.
        """,
        "formula": "(A) & (~A | B) & (~B | C) & (~C | D) & (~D | E) & (~E)",
        "expected": {
            "satisfiable": False,
            "explanation": "A=T forces B=T, C=T, D=T, E=T, but ~E requires E=F"
        },
        "why_hard": "Requires tracking 5-step implication chain"
    },
    
    # -------------------------------------------------------------------------
    # Test 3: XOR ladder (parity problem)
    # -------------------------------------------------------------------------
    {
        "id": "sat_xor_4",
        "name": "4-Variable XOR Chain",
        "difficulty": "hard",
        "description": """
        XOR constraints create parity requirements that are hard to track.
        A⊕B, B⊕C, C⊕D, with specific boundary conditions.
        """,
        "formula": "(A | B) & (~A | ~B) & (B | C) & (~B | ~C) & (C | D) & (~C | ~D) & (A) & (~D)",
        "expected": {
            "satisfiable": False,
            "explanation": "XOR chain with A=T requires D=T, contradicts ~D"
        },
        "why_hard": "Parity/XOR reasoning is notoriously hard for neural nets"
    },
    
    # -------------------------------------------------------------------------
    # Test 4: Resolution-heavy UNSAT
    # -------------------------------------------------------------------------
    {
        "id": "sat_resolution_hard",
        "name": "Resolution-Heavy UNSAT",
        "difficulty": "very_hard",
        "description": """
        Requires multiple resolution steps to derive contradiction.
        No single variable assignment immediately fails.
        """,
        "formula": "(A | B | C) & (A | B | ~C) & (A | ~B | C) & (A | ~B | ~C) & (~A | B | C) & (~A | B | ~C) & (~A | ~B | C) & (~A | ~B | ~C)",
        "expected": {
            "satisfiable": False,
            "explanation": "All 8 combinations of 3 variables - always false"
        },
        "why_hard": "Must systematically try all 8 combinations or recognize pattern"
    },

    # -------------------------------------------------------------------------
    # Test 5: Pigeonhole 4 into 3
    # -------------------------------------------------------------------------
    {
        "id": "sat_pigeonhole_4_3",
        "name": "Pigeonhole 4→3",
        "difficulty": "very_hard",
        "description": """
        4 pigeons, 3 holes. Variables Pij = pigeon i in hole j.
        Each pigeon must be somewhere, no two pigeons in same hole.
        Classic hard problem for resolution-based reasoning.
        """,
        "formula": """
        (P11 | P12 | P13) & (P21 | P22 | P23) & (P31 | P32 | P33) & (P41 | P42 | P43) &
        (~P11 | ~P21) & (~P11 | ~P31) & (~P11 | ~P41) & (~P21 | ~P31) & (~P21 | ~P41) & (~P31 | ~P41) &
        (~P12 | ~P22) & (~P12 | ~P32) & (~P12 | ~P42) & (~P22 | ~P32) & (~P22 | ~P42) & (~P32 | ~P42) &
        (~P13 | ~P23) & (~P13 | ~P33) & (~P13 | ~P43) & (~P23 | ~P33) & (~P23 | ~P43) & (~P33 | ~P43)
        """,
        "expected": {
            "satisfiable": False,
            "explanation": "Pigeonhole principle - 4 items can't fit in 3 slots uniquely"
        },
        "why_hard": "22 clauses, 12 variables, exponential search space"
    },

    # -------------------------------------------------------------------------
    # Test 6: Subtle SAT (actually satisfiable but tricky)
    # -------------------------------------------------------------------------
    {
        "id": "sat_subtle_sat",
        "name": "Subtle Satisfiable",
        "difficulty": "hard",
        "description": """
        Looks very constrained but has exactly one solution.
        Native reasoning often gives up or guesses wrong.
        """,
        "formula": "(A | B) & (~A | C) & (~B | D) & (~C | ~D) & (C | D) & (~A | ~B)",
        "expected": {
            "satisfiable": True,
            "valid_assignments": [{"A": False, "B": True, "C": False, "D": True}],
            "explanation": "Only B=T, D=T, A=F, C=F works"
        },
        "why_hard": "Many partial assignments fail, requires systematic search"
    },

    # -------------------------------------------------------------------------
    # Test 7: 6-variable interlocked
    # -------------------------------------------------------------------------
    {
        "id": "sat_interlocked_6",
        "name": "6-Variable Interlocked",
        "difficulty": "very_hard",
        "description": """
        Each variable appears in clauses with every other variable.
        Creates dense constraint graph with subtle solution.
        """,
        "formula": "(A | B) & (~A | C) & (B | ~C) & (~B | D) & (C | ~D) & (~C | E) & (D | ~E) & (~D | F) & (E | ~F) & (~E | A) & (F | ~A) & (~F | B)",
        "expected": {
            "satisfiable": True,
            "explanation": "Cyclic constraints with specific phase pattern"
        },
        "why_hard": "12 clauses forming a cycle, must track phase propagation"
    },
]

# =============================================================================
# CATEGORY 2: KNOWLEDGE GRAPH PROBLEMS
# =============================================================================

KG_HARD_TESTS = [
    # -------------------------------------------------------------------------
    # Test 1: Diamond inheritance pattern
    # -------------------------------------------------------------------------
    {
        "id": "kg_diamond",
        "name": "Diamond Pattern",
        "difficulty": "medium",
        "description": """
        Classic diamond inheritance: A→B, A→C, B→D, C→D.
        Must not double-count the A→D path.
        """,
        "edges": [
            ["A", "parent_of", "B"],
            ["A", "parent_of", "C"],
            ["B", "parent_of", "D"],
            ["C", "parent_of", "D"]
        ],
        "expected": {
            "total_edges": 5,
            "inferred_edges": [["A", "D"]],
            "note": "A→D via B and A→D via C collapse to single edge"
        },
        "why_hard": "Easy to double-count diamond paths"
    },

    # -------------------------------------------------------------------------
    # Test 2: Multi-level tree (depth 4, branching 2)
    # -------------------------------------------------------------------------
    {
        "id": "kg_tree_4_2",
        "name": "Binary Tree Depth 4",
        "difficulty": "hard",
        "description": """
        Complete binary tree: 15 nodes, 14 edges.
        Must compute all ancestor relationships.
        """,
        "edges": [
            ["N1", "parent_of", "N2"], ["N1", "parent_of", "N3"],
            ["N2", "parent_of", "N4"], ["N2", "parent_of", "N5"],
            ["N3", "parent_of", "N6"], ["N3", "parent_of", "N7"],
            ["N4", "parent_of", "N8"], ["N4", "parent_of", "N9"],
            ["N5", "parent_of", "N10"], ["N5", "parent_of", "N11"],
            ["N6", "parent_of", "N12"], ["N6", "parent_of", "N13"],
            ["N7", "parent_of", "N14"], ["N7", "parent_of", "N15"]
        ],
        "expected": {
            "direct_count": 14,
            "total_edges": 49,
            "breakdown": {
                "N1_reaches": 14,  # all descendants
                "N2_reaches": 6,   # N4,N5,N8,N9,N10,N11
                "N3_reaches": 6,   # N6,N7,N12,N13,N14,N15
                "N4_reaches": 2,   # N8,N9
                "N5_reaches": 2,   # N10,N11
                "N6_reaches": 2,   # N12,N13
                "N7_reaches": 2,   # N14,N15
                "leaves_reach": 0
            }
        },
        "why_hard": "Must systematically compute 49 edges without missing any"
    },

    # -------------------------------------------------------------------------
    # Test 3: DAG with convergence
    # -------------------------------------------------------------------------
    {
        "id": "kg_dag_converge",
        "name": "Converging DAG",
        "difficulty": "very_hard",
        "description": """
        Multiple entry points converging to single sink.
        Complex path counting.
        """,
        "edges": [
            ["A", "flows_to", "D"],
            ["B", "flows_to", "D"],
            ["C", "flows_to", "E"],
            ["D", "flows_to", "F"],
            ["E", "flows_to", "F"],
            ["D", "flows_to", "G"],
            ["E", "flows_to", "G"],
            ["F", "flows_to", "H"],
            ["G", "flows_to", "H"],
            ["A", "flows_to", "E"],
            ["B", "flows_to", "F"]
        ],
        "expected": {
            "direct_count": 11,
            "total_edges": 26,
            "key_inferences": [
                "A→F (via D)", "A→G (via D)", "A→H (via D→F, D→G, E→F, E→G)",
                "B→G (via D)", "B→H (via D, F)",
                "C→F (via E)", "C→G (via E)", "C→H (via E)"
            ]
        },
        "why_hard": "Multiple paths to same node, must track all without duplicates"
    },

    # -------------------------------------------------------------------------
    # Test 4: Long chain with branches
    # -------------------------------------------------------------------------
    {
        "id": "kg_chain_branch",
        "name": "Chain with Branches",
        "difficulty": "hard",
        "description": """
        Main chain A→B→C→D→E with branches at each node.
        Tests systematic traversal.
        """,
        "edges": [
            ["A", "parent_of", "B"], ["A", "parent_of", "A1"],
            ["B", "parent_of", "C"], ["B", "parent_of", "B1"],
            ["C", "parent_of", "D"], ["C", "parent_of", "C1"],
            ["D", "parent_of", "E"], ["D", "parent_of", "D1"],
            ["E", "parent_of", "E1"]
        ],
        "expected": {
            "direct_count": 9,
            "total_edges": 24,
            "explanation": """
            A reaches: B,A1,C,B1,D,C1,E,D1,E1 (9)
            B reaches: C,B1,D,C1,E,D1,E1 (7)
            C reaches: D,C1,E,D1,E1 (5)
            D reaches: E,D1,E1 (3)
            E reaches: E1 (1)
            Total: 9+7+5+3+0 = 24 (branches don't reach further)
            """
        },
        "why_hard": "Must track both chain propagation and branch endpoints"
    },

    # -------------------------------------------------------------------------
    # Test 5: Multiple relation types (if supported)
    # -------------------------------------------------------------------------
    {
        "id": "kg_multi_relation",
        "name": "Multi-Relation Graph",
        "difficulty": "hard",
        "description": """
        Same nodes, different relations. Tests if closure is per-relation.
        """,
        "edges": [
            ["Alice", "manages", "Bob"],
            ["Bob", "manages", "Carol"],
            ["Alice", "mentors", "Carol"],
            ["Carol", "mentors", "Dave"],
            ["Bob", "collaborates", "Dave"]
        ],
        "expected": {
            "per_relation_closure": {
                "manages": {"total": 3, "inferred": [["Alice", "Carol"]]},
                "mentors": {"total": 3, "inferred": [["Alice", "Dave"]]},
                "collaborates": {"total": 1, "inferred": []}
            }
        },
        "why_hard": "Must separate relation types in closure computation"
    },
]

# =============================================================================
# CATEGORY 3: CONSTRAINT SATISFACTION PROBLEMS
# =============================================================================

CSP_HARD_TESTS = [
    # -------------------------------------------------------------------------
    # Test 1: 8-Queens
    # -------------------------------------------------------------------------
    {
        "id": "csp_8queens",
        "name": "8-Queens Problem",
        "difficulty": "very_hard",
        "description": """
        Place 8 queens on 8x8 board. No two attack each other.
        8 variables (column positions), 8 values each, 56 constraints.
        """,
        "variables": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"],
        "domains": {f"Q{i}": list(range(1, 9)) for i in range(1, 9)},
        "constraints": [
            # All different rows
            "Q1 != Q2", "Q1 != Q3", "Q1 != Q4", "Q1 != Q5", "Q1 != Q6", "Q1 != Q7", "Q1 != Q8",
            "Q2 != Q3", "Q2 != Q4", "Q2 != Q5", "Q2 != Q6", "Q2 != Q7", "Q2 != Q8",
            "Q3 != Q4", "Q3 != Q5", "Q3 != Q6", "Q3 != Q7", "Q3 != Q8",
            "Q4 != Q5", "Q4 != Q6", "Q4 != Q7", "Q4 != Q8",
            "Q5 != Q6", "Q5 != Q7", "Q5 != Q8",
            "Q6 != Q7", "Q6 != Q8",
            "Q7 != Q8",
            # No diagonal attacks
            "abs(Q1-Q2) != 1", "abs(Q1-Q3) != 2", "abs(Q1-Q4) != 3", "abs(Q1-Q5) != 4",
            "abs(Q1-Q6) != 5", "abs(Q1-Q7) != 6", "abs(Q1-Q8) != 7",
            "abs(Q2-Q3) != 1", "abs(Q2-Q4) != 2", "abs(Q2-Q5) != 3",
            "abs(Q2-Q6) != 4", "abs(Q2-Q7) != 5", "abs(Q2-Q8) != 6",
            "abs(Q3-Q4) != 1", "abs(Q3-Q5) != 2", "abs(Q3-Q6) != 3",
            "abs(Q3-Q7) != 4", "abs(Q3-Q8) != 5",
            "abs(Q4-Q5) != 1", "abs(Q4-Q6) != 2", "abs(Q4-Q7) != 3", "abs(Q4-Q8) != 4",
            "abs(Q5-Q6) != 1", "abs(Q5-Q7) != 2", "abs(Q5-Q8) != 3",
            "abs(Q6-Q7) != 1", "abs(Q6-Q8) != 2",
            "abs(Q7-Q8) != 1"
        ],
        "expected": {
            "satisfiable": True,
            "num_solutions": 92,
            "example_solutions": [
                {"Q1": 1, "Q2": 5, "Q3": 8, "Q4": 6, "Q5": 3, "Q6": 7, "Q7": 2, "Q8": 4},
                {"Q1": 1, "Q2": 6, "Q3": 8, "Q4": 3, "Q5": 7, "Q6": 4, "Q7": 2, "Q8": 5}
            ]
        },
        "why_hard": "8^8 = 16M search space, 56 constraints to check"
    },

    # -------------------------------------------------------------------------
    # Test 2: SEND + MORE = MONEY (Cryptarithmetic)
    # -------------------------------------------------------------------------
    {
        "id": "csp_send_more_money",
        "name": "SEND + MORE = MONEY",
        "difficulty": "very_hard",
        "description": """
        Classic cryptarithmetic puzzle.
        8 variables, each a distinct digit 0-9.
        Leading digits S, M cannot be 0.
        """,
        "variables": ["S", "E", "N", "D", "M", "O", "R", "Y"],
        "domains": {v: list(range(0, 10)) for v in ["S", "E", "N", "D", "M", "O", "R", "Y"]},
        "constraints": [
            # All different
            "S != E", "S != N", "S != D", "S != M", "S != O", "S != R", "S != Y",
            "E != N", "E != D", "E != M", "E != O", "E != R", "E != Y",
            "N != D", "N != M", "N != O", "N != R", "N != Y",
            "D != M", "D != O", "D != R", "D != Y",
            "M != O", "M != R", "M != Y",
            "O != R", "O != Y",
            "R != Y",
            # Leading digits non-zero
            "S > 0",
            "M > 0",
            # The equation: SEND + MORE = MONEY
            "1000*S + 100*E + 10*N + D + 1000*M + 100*O + 10*R + E == 10000*M + 1000*O + 100*N + 10*E + Y"
        ],
        "expected": {
            "satisfiable": True,
            "unique_solution": {"S": 9, "E": 5, "N": 6, "D": 7, "M": 1, "O": 0, "R": 8, "Y": 2},
            "verification": "9567 + 1085 = 10652"
        },
        "why_hard": "10^8 naive search space, complex arithmetic constraint"
    },

    # -------------------------------------------------------------------------
    # Test 3: Graph 3-Coloring (Petersen Graph)
    # -------------------------------------------------------------------------
    {
        "id": "csp_3color_petersen",
        "name": "Petersen Graph 3-Coloring",
        "difficulty": "very_hard",
        "description": """
        Color the Petersen graph with 3 colors.
        10 nodes, 15 edges, famous for being non-planar.
        Actually requires exactly 3 colors (chromatic number = 3).
        """,
        "variables": ["V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9"],
        "domains": {f"V{i}": [1, 2, 3] for i in range(10)},
        "constraints": [
            # Outer pentagon: 0-1-2-3-4-0
            "V0 != V1", "V1 != V2", "V2 != V3", "V3 != V4", "V4 != V0",
            # Inner pentagram: 5-7-9-6-8-5 
            "V5 != V7", "V7 != V9", "V9 != V6", "V6 != V8", "V8 != V5",
            # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
            "V0 != V5", "V1 != V6", "V2 != V7", "V3 != V8", "V4 != V9"
        ],
        "expected": {
            "satisfiable": True,
            "example_solution": {
                "V0": 1, "V1": 2, "V2": 1, "V3": 2, "V4": 3,
                "V5": 2, "V6": 1, "V7": 3, "V8": 1, "V9": 2
            }
        },
        "why_hard": "Non-planar graph, pentagram structure creates tricky constraints"
    },

    # -------------------------------------------------------------------------
    # Test 4: Magic Square 3x3
    # -------------------------------------------------------------------------
    {
        "id": "csp_magic_square_3",
        "name": "3x3 Magic Square",
        "difficulty": "hard",
        "description": """
        Fill 3x3 grid with 1-9, all rows/cols/diagonals sum to 15.
        9 variables, 9 values, 8 sum constraints + all-different.
        """,
        "variables": ["C11", "C12", "C13", "C21", "C22", "C23", "C31", "C32", "C33"],
        "domains": {f"C{i}{j}": list(range(1, 10)) for i in range(1, 4) for j in range(1, 4)},
        "constraints": [
            # All different (36 constraints)
            "C11 != C12", "C11 != C13", "C11 != C21", "C11 != C22", "C11 != C23",
            "C11 != C31", "C11 != C32", "C11 != C33",
            "C12 != C13", "C12 != C21", "C12 != C22", "C12 != C23",
            "C12 != C31", "C12 != C32", "C12 != C33",
            "C13 != C21", "C13 != C22", "C13 != C23",
            "C13 != C31", "C13 != C32", "C13 != C33",
            "C21 != C22", "C21 != C23", "C21 != C31", "C21 != C32", "C21 != C33",
            "C22 != C23", "C22 != C31", "C22 != C32", "C22 != C33",
            "C23 != C31", "C23 != C32", "C23 != C33",
            "C31 != C32", "C31 != C33",
            "C32 != C33",
            # Rows sum to 15
            "C11 + C12 + C13 == 15",
            "C21 + C22 + C23 == 15",
            "C31 + C32 + C33 == 15",
            # Cols sum to 15
            "C11 + C21 + C31 == 15",
            "C12 + C22 + C32 == 15",
            "C13 + C23 + C33 == 15",
            # Diagonals sum to 15
            "C11 + C22 + C33 == 15",
            "C13 + C22 + C31 == 15"
        ],
        "expected": {
            "satisfiable": True,
            "num_solutions": 8,  # 1 unique + rotations/reflections
            "canonical_solution": {
                "C11": 2, "C12": 7, "C13": 6,
                "C21": 9, "C22": 5, "C23": 1,
                "C31": 4, "C32": 3, "C33": 8
            }
        },
        "why_hard": "9! = 362880 permutations, 8 sum constraints must align"
    },

    # -------------------------------------------------------------------------
    # Test 5: Scheduling Problem
    # -------------------------------------------------------------------------
    {
        "id": "csp_scheduling",
        "name": "Job Shop Scheduling",
        "difficulty": "hard",
        "description": """
        4 jobs, 3 machines. Each job visits each machine once.
        Minimize makespan (or just find feasible schedule).
        No two jobs on same machine at same time.
        """,
        "variables": [
            # Start times for each job on each machine
            "J1M1", "J1M2", "J1M3",
            "J2M1", "J2M2", "J2M3",
            "J3M1", "J3M2", "J3M3",
            "J4M1", "J4M2", "J4M3"
        ],
        "domains": {v: list(range(0, 12)) for v in [
            "J1M1", "J1M2", "J1M3", "J2M1", "J2M2", "J2M3",
            "J3M1", "J3M2", "J3M3", "J4M1", "J4M2", "J4M3"
        ]},
        "constraints": [
            # Job order constraints (each job: M1 → M2 → M3, duration 1 each)
            "J1M1 + 1 <= J1M2", "J1M2 + 1 <= J1M3",
            "J2M1 + 1 <= J2M2", "J2M2 + 1 <= J2M3",
            "J3M1 + 1 <= J3M2", "J3M2 + 1 <= J3M3",
            "J4M1 + 1 <= J4M2", "J4M2 + 1 <= J4M3",
            # Machine capacity (no overlap) - disjunctive constraints
            # Machine 1
            "(J1M1 + 1 <= J2M1) | (J2M1 + 1 <= J1M1)",
            "(J1M1 + 1 <= J3M1) | (J3M1 + 1 <= J1M1)",
            "(J1M1 + 1 <= J4M1) | (J4M1 + 1 <= J1M1)",
            "(J2M1 + 1 <= J3M1) | (J3M1 + 1 <= J2M1)",
            "(J2M1 + 1 <= J4M1) | (J4M1 + 1 <= J2M1)",
            "(J3M1 + 1 <= J4M1) | (J4M1 + 1 <= J3M1)",
            # Machine 2
            "(J1M2 + 1 <= J2M2) | (J2M2 + 1 <= J1M2)",
            "(J1M2 + 1 <= J3M2) | (J3M2 + 1 <= J1M2)",
            "(J1M2 + 1 <= J4M2) | (J4M2 + 1 <= J1M2)",
            "(J2M2 + 1 <= J3M2) | (J3M2 + 1 <= J2M2)",
            "(J2M2 + 1 <= J4M2) | (J4M2 + 1 <= J2M2)",
            "(J3M2 + 1 <= J4M2) | (J4M2 + 1 <= J3M2)",
            # Machine 3
            "(J1M3 + 1 <= J2M3) | (J2M3 + 1 <= J1M3)",
            "(J1M3 + 1 <= J3M3) | (J3M3 + 1 <= J1M3)",
            "(J1M3 + 1 <= J4M3) | (J4M3 + 1 <= J1M3)",
            "(J2M3 + 1 <= J3M3) | (J3M3 + 1 <= J2M3)",
            "(J2M3 + 1 <= J4M3) | (J4M3 + 1 <= J2M3)",
            "(J3M3 + 1 <= J4M3) | (J4M3 + 1 <= J3M3)"
        ],
        "expected": {
            "satisfiable": True,
            "example_solution": {
                "J1M1": 0, "J1M2": 1, "J1M3": 2,
                "J2M1": 1, "J2M2": 2, "J2M3": 3,
                "J3M1": 2, "J3M2": 3, "J3M3": 4,
                "J4M1": 3, "J4M2": 4, "J4M3": 5
            }
        },
        "why_hard": "Disjunctive constraints require branching on orderings"
    },

    # -------------------------------------------------------------------------
    # Test 6: Impossible CSP (Overconstrained)
    # -------------------------------------------------------------------------
    {
        "id": "csp_impossible_hard",
        "name": "Overconstrained UNSAT",
        "difficulty": "hard",
        "description": """
        Looks like it might have a solution but mathematically impossible.
        Tests whether solver correctly identifies UNSAT.
        """,
        "variables": ["A", "B", "C", "D"],
        "domains": {"A": [1, 2, 3], "B": [1, 2, 3], "C": [1, 2, 3], "D": [1, 2, 3]},
        "constraints": [
            "A + B + C + D == 14",  # Max sum = 12 (3+3+3+3)
            "A != B", "B != C", "C != D", "A != D"
        ],
        "expected": {
            "satisfiable": False,
            "explanation": "Sum constraint requires 14, max possible is 12"
        },
        "why_hard": "Sum constraint is subtly impossible given domains"
    },

    # -------------------------------------------------------------------------
    # Test 7: Sudoku (partial - 4x4)
    # -------------------------------------------------------------------------
    {
        "id": "csp_sudoku_4x4",
        "name": "4x4 Sudoku",
        "difficulty": "hard",
        "description": """
        4x4 Sudoku: fill grid with 1-4, each row/col/2x2 box has all digits.
        Some cells pre-filled.
        """,
        "variables": ["C11", "C12", "C13", "C14",
                      "C21", "C22", "C23", "C24",
                      "C31", "C32", "C33", "C34",
                      "C41", "C42", "C43", "C44"],
        "domains": {
            # Pre-filled cells have single value
            "C11": [1], "C14": [4],
            "C22": [3], "C23": [1],
            "C32": [4], "C33": [2],
            "C41": [2], "C44": [3],
            # Empty cells have full domain
            "C12": [1, 2, 3, 4], "C13": [1, 2, 3, 4],
            "C21": [1, 2, 3, 4], "C24": [1, 2, 3, 4],
            "C31": [1, 2, 3, 4], "C34": [1, 2, 3, 4],
            "C42": [1, 2, 3, 4], "C43": [1, 2, 3, 4]
        },
        "constraints": [
            # Row constraints
            "C11 != C12", "C11 != C13", "C11 != C14", "C12 != C13", "C12 != C14", "C13 != C14",
            "C21 != C22", "C21 != C23", "C21 != C24", "C22 != C23", "C22 != C24", "C23 != C24",
            "C31 != C32", "C31 != C33", "C31 != C34", "C32 != C33", "C32 != C34", "C33 != C34",
            "C41 != C42", "C41 != C43", "C41 != C44", "C42 != C43", "C42 != C44", "C43 != C44",
            # Column constraints
            "C11 != C21", "C11 != C31", "C11 != C41", "C21 != C31", "C21 != C41", "C31 != C41",
            "C12 != C22", "C12 != C32", "C12 != C42", "C22 != C32", "C22 != C42", "C32 != C42",
            "C13 != C23", "C13 != C33", "C13 != C43", "C23 != C33", "C23 != C43", "C33 != C43",
            "C14 != C24", "C14 != C34", "C14 != C44", "C24 != C34", "C24 != C44", "C34 != C44",
            # Box constraints (2x2)
            "C11 != C22", "C12 != C21",  # Top-left box
            "C13 != C24", "C14 != C23",  # Top-right box
            "C31 != C42", "C32 != C41",  # Bottom-left box
            "C33 != C44", "C34 != C43"   # Bottom-right box
        ],
        "expected": {
            "satisfiable": True,
            "solution": {
                "C11": 1, "C12": 2, "C13": 3, "C14": 4,
                "C21": 4, "C22": 3, "C23": 1, "C24": 2,
                "C31": 3, "C32": 4, "C33": 2, "C34": 1,
                "C41": 2, "C42": 1, "C43": 4, "C44": 3
            }
        },
        "why_hard": "48 constraints, must propagate pre-filled values correctly"
    },
]


# =============================================================================
# NATIVE REASONING PROMPTS
# =============================================================================

def generate_native_prompts():
    """Generate prompts for testing native Claude reasoning."""
    
    prompts = []
    
    # SAT prompts
    for test in SAT_HARD_TESTS:
        formula = test["formula"].replace("\n", " ").strip()
        prompts.append({
            "id": test["id"],
            "type": "SAT",
            "prompt": f"""Solve this SAT problem WITHOUT using any tools. Show your complete reasoning.

Formula: {formula}

Determine if this is SATISFIABLE or UNSATISFIABLE.
If satisfiable, provide a valid truth assignment for all variables.
Show your work step by step.""",
            "expected": test["expected"]
        })
    
    # KG prompts
    for test in KG_HARD_TESTS:
        edges_str = "\n".join([f"  {e[0]} → {e[2]}" for e in test["edges"]])
        prompts.append({
            "id": test["id"],
            "type": "KG",
            "prompt": f"""Compute the transitive closure of this graph WITHOUT using any tools.

Direct edges:
{edges_str}

1. List ALL direct edges (the ones given above)
2. List ALL inferred edges (reachable via transitivity)
3. Give the EXACT total count of edges in the closure

Be systematic. Do not miss any edges. Show your work.""",
            "expected": test["expected"]
        })
    
    # CSP prompts  
    for test in CSP_HARD_TESTS:
        constraints_str = "\n".join([f"  {c}" for c in test["constraints"][:20]])
        if len(test["constraints"]) > 20:
            constraints_str += f"\n  ... and {len(test['constraints']) - 20} more constraints"
        
        prompts.append({
            "id": test["id"],
            "type": "CSP",
            "prompt": f"""Solve this constraint satisfaction problem WITHOUT using any tools.

Variables: {test["variables"]}
Domains: {test["domains"]}

Constraints:
{constraints_str}

Find a valid assignment that satisfies ALL constraints, or prove no solution exists.
Show your systematic search process.""",
            "expected": test["expected"]
        })
    
    return prompts


# =============================================================================
# SUMMARY
# =============================================================================

def print_test_summary():
    """Print summary of all hard tests."""
    print("=" * 70)
    print("REASONFORGE HARD TEST SUITE")
    print("=" * 70)
    
    print(f"\nSAT Problems: {len(SAT_HARD_TESTS)}")
    for t in SAT_HARD_TESTS:
        exp = "UNSAT" if not t["expected"].get("satisfiable", True) else "SAT"
        print(f"  [{t['difficulty']:10}] {t['id']:25} → {exp}")
    
    print(f"\nKnowledge Graph Problems: {len(KG_HARD_TESTS)}")
    for t in KG_HARD_TESTS:
        edges = t["expected"].get("total_edges", "?")
        print(f"  [{t['difficulty']:10}] {t['id']:25} → {edges} edges")
    
    print(f"\nCSP Problems: {len(CSP_HARD_TESTS)}")
    for t in CSP_HARD_TESTS:
        exp = "UNSAT" if not t["expected"].get("satisfiable", True) else "SAT"
        nvars = len(t["variables"])
        ncons = len(t["constraints"])
        print(f"  [{t['difficulty']:10}] {t['id']:25} → {exp} ({nvars} vars, {ncons} constraints)")
    
    total = len(SAT_HARD_TESTS) + len(KG_HARD_TESTS) + len(CSP_HARD_TESTS)
    print(f"\nTotal: {total} hard test cases")
    print("=" * 70)


if __name__ == "__main__":
    print_test_summary()
    print("\n\nGenerating native reasoning prompts...\n")
    
    prompts = generate_native_prompts()
    for p in prompts[:3]:  # Show first 3
        print(f"--- {p['id']} ({p['type']}) ---")
        print(p["prompt"][:500] + "...")
        print()
