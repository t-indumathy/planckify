"""100 diverse prompts used by both Locust load-test flavours.

Categories: physics, chemistry, biology, math, programming, AI/ML,
history, geography, economics, literature, technology, logic, environment,
medicine, space, practical engineering.
"""

PROMPTS: list[str] = [
    # ── Physics ────────────────────────────────────────────────────────────
    "What is the Planck constant and why is it important in physics?",
    "Explain quantum entanglement in simple terms.",
    "How does Einstein's special theory of relativity change our understanding of time?",
    "What is dark matter and why is it hard to detect?",
    "Describe the double-slit experiment and what it reveals about light.",
    "What is the Heisenberg uncertainty principle?",
    "How do nuclear reactors generate electricity?",
    "Explain the electromagnetic spectrum from radio waves to gamma rays.",
    "What is entropy in thermodynamics?",
    "How does a black hole form from a dying star?",
    # ── Chemistry ──────────────────────────────────────────────────────────
    "Explain the difference between covalent and ionic bonding.",
    "What is the difference between an acid and a base?",
    "How does the periodic table organise the elements?",
    "What is polymerisation and give a real-world example.",
    "How does photosynthesis work at a chemical level?",
    # ── Biology ────────────────────────────────────────────────────────────
    "How does DNA replication work?",
    "Explain the process of natural selection.",
    "What is the difference between mitosis and meiosis?",
    "How do vaccines train the immune system?",
    "Describe the structure and function of a neuron.",
    # ── Mathematics ────────────────────────────────────────────────────────
    "Prove that the square root of 2 is irrational.",
    "Explain Euler's identity e^(iπ) + 1 = 0.",
    "What is the difference between a permutation and a combination?",
    "Describe the concept of a derivative in calculus.",
    "What is the travelling salesman problem and why is it hard?",
    # ── Programming / Computer Science ─────────────────────────────────────
    "Write a Python function to check if a number is prime.",
    "Explain the difference between a stack and a queue.",
    "What is the time complexity of quicksort in the average case?",
    "How does garbage collection work in Python?",
    "Explain the concept of recursion with a clear example.",
    "What is the difference between HTTP and HTTPS?",
    "How does a hash map work internally?",
    "Explain the CAP theorem in distributed systems.",
    "What is dependency injection and when should you use it?",
    "How does version control with git work at a high level?",
    # ── AI / ML ────────────────────────────────────────────────────────────
    "What are the main differences between supervised and unsupervised learning?",
    "How does a neural network learn from data through backpropagation?",
    "Explain the attention mechanism in transformer models.",
    "What is the difference between ONNX and TensorFlow Lite model formats?",
    "How does dropout regularisation prevent overfitting?",
    "What is quantisation in machine learning and why does it matter for edge devices?",
    "Explain gradient descent and its variants.",
    "What is a convolutional neural network and what is it good at?",
    "How does BERT differ from GPT architecturally?",
    "What is reinforcement learning and where is it applied?",
    # ── History ────────────────────────────────────────────────────────────
    "What were the main causes of World War I?",
    "Explain the significance of the French Revolution for modern politics.",
    "What was the impact of the Industrial Revolution on society?",
    "Describe the fall of the Western Roman Empire.",
    "What were the causes and consequences of the Cold War?",
    # ── Geography / Earth Science ──────────────────────────────────────────
    "Explain the water cycle in detail.",
    "What causes earthquakes and how are they measured?",
    "How do ocean currents affect global climate?",
    "What is the Ring of Fire and why is it geologically active?",
    "How does plate tectonics and continental drift work?",
    # ── Economics ──────────────────────────────────────────────────────────
    "Explain the law of supply and demand.",
    "What is GDP and how is it measured?",
    "How does inflation affect an economy?",
    "What is comparative advantage in international trade?",
    "Explain the concept of opportunity cost with an example.",
    # ── Literature / Writing ───────────────────────────────────────────────
    "Write a short poem about the night sky.",
    "Explain the hero's journey narrative structure.",
    "What makes a compelling opening paragraph for a short story?",
    "Describe the main themes of Shakespeare's Hamlet.",
    "How does the use of metaphor enhance writing?",
    # ── Technology ─────────────────────────────────────────────────────────
    "How does 5G differ from 4G in terms of technology?",
    "What is blockchain technology and how does it achieve trust?",
    "How do solid-state drives (SSDs) work compared to hard drives?",
    "Explain how the internet routes data packets from source to destination.",
    "What is quantum computing and why does it matter for cryptography?",
    # ── Logic & Philosophy ─────────────────────────────────────────────────
    "What is the Monty Hall problem and what is the correct solution?",
    "Explain the trolley problem in ethics.",
    "What is Occam's razor and when should it be applied?",
    "Describe Russell's paradox in set theory.",
    "What is Gödel's incompleteness theorem?",
    # ── Environment ────────────────────────────────────────────────────────
    "How does the greenhouse effect work?",
    "What is the difference between climate and weather?",
    "Explain how solar and wind renewable energy sources work.",
    "What is biodiversity and why does it matter for ecosystems?",
    "How does ocean acidification happen and what are its effects?",
    # ── Medicine / Health ──────────────────────────────────────────────────
    "How does the immune system fight a bacterial infection?",
    "What is CRISPR-Cas9 gene editing and how does it work?",
    "How do antibiotics work to kill bacteria?",
    "Explain how mRNA vaccines are designed and manufactured.",
    "What is the difference between bacteria and viruses?",
    # ── Space / Cosmology ──────────────────────────────────────────────────
    "How are stars formed from nebulae?",
    "What is a neutron star and what are its extreme properties?",
    "Explain the Big Bang theory and the evidence supporting it.",
    "How does gravitational lensing work?",
    "What is the Drake equation and what does it estimate?",
    # ── Practical / Engineering ────────────────────────────────────────────
    "Explain how to write effective unit tests for a Python function.",
    "How do you optimise a slow SQL query?",
    "What steps would you take to debug a memory leak in a long-running process?",
    "How do you design a RESTful API?",
    "Explain the process of containerising an application with Docker.",
    "What is the difference between RAM and ROM?",
    "How does a CPU execute machine instructions?",
    "Explain floating-point precision errors with an example.",
    "What is the difference between synchronous and asynchronous programming?",
    "How do Neural Processing Units (NPUs) differ from GPUs and CPUs?",
]

assert len(PROMPTS) == 100, f"Expected 100 prompts, got {len(PROMPTS)}"
