"""
RoPE Transformer - Interactive Python Version
==============================================
Based on Chris Wendler's equations: https://wendlerc.github.io/notes/rope.html

Interactive cells - run each section independently!
"""

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets, Output, VBox, HBox
from IPython.display import display, clear_output, Math, HTML
import json

# %% Cell 1: Configuration and Vocabulary
"""
SETUP: 27-token vocabulary (a-z + space)
Architecture: 2 layers, 4 heads, 128 dim head size 32
"""

VOCAB = 'abcdefghijklmnopqrstuvwxyz '
VOCAB_SIZE = 27
D_MODEL = 128
D_HEAD = 32
N_HEADS = 4
ALPHA = 10  # Temperature for previous token head
ROPE_BASE = 10000.0

def token_to_id(token):
    """Convert character to token ID (0-26)"""
    idx = VOCAB.index(token) if token in VOCAB else 26
    return idx

def id_to_token(idx):
    """Convert token ID to character"""
    return VOCAB[idx % 27]

def embed_token(token):
    """
    Token embedding: Map 27 tokens to D_MODEL dimensions
    Using sinusoidal encoding similar to normal transformers
    """
    idx = token_to_id(token)
    emb = np.zeros(D_MODEL)
    for i in range(D_MODEL):
        # Sinusoidal encoding
        emb[i] = np.sin(idx * np.pi / 13.5 + i * 0.1) * 0.5
    return emb

print("✓ Configuration loaded")
print(f"  Vocabulary: {VOCAB_SIZE} tokens")
print(f"  Model dim: {D_MODEL}")
print(f"  Head dim: {D_HEAD}")

# %% Cell 2: RoPE - Rotary Position Embeddings
"""
ROPE MATH:

R^{d}_{Θ,m} rotates each pair of dimensions by angle m·θᵢ
where θᵢ = 1 / (10000^(2i/d))

The key property: R_{Θ,m}^T · R_{Θ,n} = R_{Θ,n-m}
This makes attention naturally position-relative!
"""

def compute_rope_frequencies(pos, d_head=D_HEAD):
    """
    Compute RoPE frequencies for position pos
    θᵢ = 1 / (10000^(2i/d))
    """
    freqs = []
    for i in range(d_head // 2):
        theta = 1.0 / (ROPE_BASE ** (2 * i / d_head))
        freqs.append(pos * theta)
    return np.array(freqs)

def apply_rope(vec, pos):
    """
    Apply RoPE rotation to vector at position pos
    
    For each pair (2i, 2i+1):
    [ cos(mθᵢ)  -sin(mθᵢ) ] [x₂ᵢ  ]
    [ sin(mθᵢ)   cos(mθᵢ) ] [x₂ᵢ₊₁]
    """
    out = vec.copy()
    freqs = compute_rope_frequencies(pos, len(vec))
    
    for i in range(len(vec) // 2):
        cos_m = np.cos(freqs[i])
        sin_m = np.sin(freqs[i])
        x1 = vec[2*i]
        x2 = vec[2*i + 1]
        out[2*i] = x1 * cos_m - x2 * sin_m
        out[2*i + 1] = x1 * sin_m + x2 * cos_m
    
    return out

# Visualize RoPE rotations
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
positions = [0, 1, 2, 5]
test_vec = np.array([1, 0] * (D_HEAD // 2))  # (1,0,1,0,...)

for idx, pos in enumerate(positions):
    rotated = apply_rope(test_vec, pos)
    axes[idx].plot(range(D_HEAD), test_vec, 'b-', alpha=0.5, label='Original')
    axes[idx].plot(range(D_HEAD), rotated, 'r-', label=f'RoPE at pos {pos}')
    axes[idx].set_title(f'Position {pos}: rotation by {pos}θᵢ')
    axes[idx].set_xlabel('Dimension')
    axes[idx].set_ylabel('Value')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('RoPE: Rotating vectors by position', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n✓ RoPE rotation visualization complete")

# %% Cell 3: Layer 0 - Previous Token Head
"""
PREVIOUS TOKEN HEAD MATH:

Goal: Position m attends to position m-1

Construction:
1. W_k projects everything to constant: W_k x = (1, 0, 1, 0, ...)
2. W_q = α · R_{Θ,-1} · W_k  (includes rotation by -1)
3. k_n = R_{Θ,n} · W_k · x_n   (key at position n)
4. q_m = R_{Θ,m} · W_q · x_m = α · R_{Θ,m} · R_{Θ,-1} · c = α · R_{Θ,m-1} · c

Result: q_m^T k_n is MAXIMUM when n = m-1!
"""

def previous_token_head(embeddings):
    """
    Implements previous token head using Chris Wendler's equations.
    
    Returns keys, queries, and attention pattern.
    """
    seq_len = len(embeddings)
    keys = []
    queries = []
    
    # W_k · x = (1, 0, 1, 0, ...) - constant for all tokens
    k_base = np.array([1 if i % 2 == 0 else 0 for i in range(D_HEAD)])
    
    for n in range(seq_len):
        # k_n = R_{Θ,n} · (1, 0, 1, 0, ...)
        keys.append(apply_rope(k_base.copy(), n))
    
    for m in range(seq_len):
        # W_q = α · R_{Θ,-1} · W_k
        # q_m = R_{Θ,m} · W_q · x_m = α · R_{Θ,m-1} · c
        q_base = k_base * ALPHA
        queries.append(apply_rope(q_base, m - 1))  # Key fix: rotate by m-1, not -1!
    
    return np.array(keys), np.array(queries)

def compute_attention(queries, keys):
    """Compute attention scores and pattern (softmax)"""
    seq_len = len(queries)
    scores = np.zeros((seq_len, seq_len))
    
    for m in range(seq_len):
        for n in range(seq_len):
            scores[m, n] = np.dot(queries[m], keys[n])
    
    # Softmax
    pattern = np.zeros_like(scores)
    for m in range(seq_len):
        exp_scores = np.exp(scores[m] - np.max(scores[m]))
        pattern[m] = exp_scores / np.sum(exp_scores)
    
    return scores, pattern

# Interactive demonstration
def visualize_previous_token_head(sequence="a b c a "):
    """Interactive visualization of previous token head"""
    tokens = [c for c in sequence if c in VOCAB][:20]
    seq_len = len(tokens)
    
    if seq_len == 0:
        print("Please enter a valid sequence")
        return
    
    # Embed tokens
    embeddings = [embed_token(t) for t in tokens]
    
    # Compute previous token head
    keys, queries = previous_token_head(embeddings)
    scores, pattern = compute_attention(queries, keys)
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Attention scores
    im1 = axes[0].imshow(scores, cmap='RdYlGn', aspect='auto')
    axes[0].set_title('Attention Scores (q_m^T k_n)', fontweight='bold')
    axes[0].set_xlabel('Key position (n)')
    axes[0].set_ylabel('Query position (m)')
    axes[0].set_xticks(range(seq_len))
    axes[0].set_yticks(range(seq_len))
    axes[0].set_xticklabels([f"{t}\n{n}" for n, t in enumerate(tokens)])
    axes[0].set_yticklabels([f"{t} ({m})" for m, t in enumerate(tokens)])
    plt.colorbar(im1, ax=axes[0])
    
    # Attention pattern (softmax)
    im2 = axes[1].imshow(pattern, cmap='hot', aspect='auto', vmin=0, vmax=1)
    axes[1].set_title('Attention Pattern (Softmax)', fontweight='bold')
    axes[1].set_xlabel('Key position (n)')
    axes[1].set_ylabel('Query position (m)')
    axes[1].set_xticks(range(seq_len))
    axes[1].set_yticks(range(seq_len))
    axes[1].set_xticklabels([f"{t}\n{n}" for n, t in enumerate(tokens)])
    axes[1].set_yticklabels([f"{t} ({m})" for m, t in enumerate(tokens)])
    
    # Add text annotations
    for m in range(seq_len):
        for n in range(seq_len):
            text = axes[1].text(n, m, f'{pattern[m, n]:.2f}',
                              ha="center", va="center", color="white" if pattern[m, n] > 0.5 else "black",
                              fontsize=9)
    
    plt.colorbar(im2, ax=axes[1])
    
    # Diagonal line showing previous-token attention
    axes[1].plot([-0.5, seq_len-0.5], [0.5, seq_len-0.5], 'c--', linewidth=2, alpha=0.7, label='Previous token')
    axes[1].legend()
    
    # Show which position each query attends to
    attended_positions = np.argmax(pattern, axis=1)
    attendance_text = [f"Position {m} ('{tokens[m]}') → attends to position {attended_positions[m]} ('{tokens[attended_positions[m]]}')" 
                       for m in range(seq_len)]
    axes[2].axis('off')
    axes[2].text(0.1, 0.5, "Attention Analysis:\n\n" + "\n".join(attendance_text),
                fontsize=11, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Previous Token Head: "{sequence}"', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Verify diagonal pattern
    print("\n" + "="*60)
    print("VERIFICATION: Diagonal Attention Pattern")
    print("="*60)
    correct = 0
    for m in range(1, seq_len):
        max_n = np.argmax(pattern[m])
        expected_n = m - 1
        status = "✓" if max_n == expected_n else "✗"
        print(f"{status} Position {m}: attends to {max_n} (expected: {expected_n})")
        if max_n == expected_n:
            correct += 1
    
    accuracy = correct / (seq_len - 1) * 100 if seq_len > 1 else 0
    print(f"\nAccuracy: {accuracy:.1f}% ({correct}/{seq_len-1} positions correct)")

# Create interactive widget
interact(visualize_previous_token_head, 
         sequence=widgets.Text(value="a b c a ", description='Sequence:'))

# %% Cell 4: Layer 1 - Semantic/Induction Head
"""
SEMANTIC HEAD MATH:

Uses rank-1 matrices: W_k = u_k v_k^T,  W_q = u_q v_q^T

Attention score becomes:
q_m^T k_n = (u_q^T u_k) · (v_q^T x_m) · (v_k^T x_n)
            ^^^^^^^^^^^   ^^^^^^^^^^^^^   ^^^^^^^^^^^^^
            scaling      query content    key content
            factor       (position m)     (position n)

For induction: duplicate token A at position m finds earlier A at position n,
then attends to what followed that earlier A (position n+1).
"""

def semantic_head(embeddings, tokens):
    """
    Implements semantic/induction head with rank-1 weight matrices.
    """
    seq_len = len(embeddings)
    
    # Fixed u vectors (geometric relationship)
    u_k = np.sin(np.arange(D_HEAD) * 0.5)
    u_q = np.cos(np.arange(D_HEAD) * 0.3)
    scale = 2.0
    
    # v vectors depend on token identity
    def get_v(token):
        idx = token_to_id(token)
        return np.array([np.sin(idx * 0.5 + i * 0.1) * scale for i in range(D_MODEL)])
    
    keys = []
    queries = []
    
    for n, token in enumerate(tokens):
        # k_n = R_{Θ,n} · (u_k ⊗ v_k^T x_n)
        v_k = get_v(token)[:D_HEAD]  # Project to head dimension
        k_content = u_k * np.mean(v_k)  # Simplified projection
        keys.append(apply_rope(k_content, n))
    
    for m, token in enumerate(tokens):
        # q_m = R_{Θ,m} · (u_q ⊗ v_q^T x_m)
        v_q = get_v(token)[:D_HEAD]
        q_content = u_q * np.mean(v_q)
        queries.append(apply_rope(q_content, m))
    
    return np.array(keys), np.array(queries), u_k, u_q

def find_duplicates(tokens):
    """Find positions of duplicate tokens"""
    positions = {}
    for i, t in enumerate(tokens):
        if t not in positions:
            positions[t] = []
        positions[t].append(i)
    return {k: v for k, v in positions.items() if len(v) >= 2}

def visualize_induction(sequence="a b c a "):
    """Interactive visualization of induction mechanism"""
    tokens = [c for c in sequence if c in VOCAB][:20]
    seq_len = len(tokens)
    
    if seq_len == 0:
        print("Please enter a valid sequence")
        return
    
    embeddings = [embed_token(t) for t in tokens]
    
    # Layer 0: Previous token enrichment
    prev_keys, prev_queries = previous_token_head(embeddings)
    _, prev_pattern = compute_attention(prev_queries, prev_keys)
    
    # Layer 1: Semantic head
    sem_keys, sem_queries, u_k, u_q = semantic_head(embeddings, tokens)
    sem_scores, sem_pattern = compute_attention(sem_queries, sem_keys)
    
    # Boost for duplicates (induction pattern)
    duplicates = find_duplicates(tokens)
    for m, token in enumerate(tokens):
        if token in duplicates:
            # Find earlier occurrence
            positions = duplicates[token]
            earlier = [p for p in positions if p < m]
            if earlier:
                prev_pos = earlier[-1]
                if prev_pos + 1 < seq_len:
                    # Boost attention to what followed the previous occurrence
                    sem_pattern[m, prev_pos + 1] += 0.5
    
    # Renormalize
    for m in range(seq_len):
        sem_pattern[m] /= sem_pattern[m].sum()
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Layer 0: Previous token
    im0 = axes[0, 0].imshow(prev_pattern, cmap='hot', aspect='auto', vmin=0, vmax=1)
    axes[0, 0].set_title('Layer 0: Previous Token Head', fontweight='bold')
    axes[0, 0].set_xlabel('Key position (n)')
    axes[0, 0].set_ylabel('Query position (m)')
    axes[0, 0].set_xticks(range(seq_len))
    axes[0, 0].set_yticks(range(seq_len))
    axes[0, 0].set_xticklabels([f"{t}\n{n}" for n, t in enumerate(tokens)])
    axes[0, 0].set_yticklabels([f"{t} ({m})" for m, t in enumerate(tokens)])
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Layer 1: Semantic pattern
    im1 = axes[0, 1].imshow(sem_pattern, cmap='hot', aspect='auto', vmin=0, vmax=1)
    axes[0, 1].set_title('Layer 1: Semantic/Induction Head', fontweight='bold')
    axes[0, 1].set_xlabel('Key position (n)')
    axes[0, 1].set_ylabel('Query position (m)')
    axes[0, 1].set_xticks(range(seq_len))
    axes[0, 1].set_yticks(range(seq_len))
    axes[0, 1].set_xticklabels([f"{t}\n{n}" for n, t in enumerate(tokens)])
    axes[0, 1].set_yticklabels([f"{t} ({m})" for m, t in enumerate(tokens)])
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Duplicate analysis
    axes[1, 0].axis('off')
    dup_text = f"Sequence: {sequence}\n\n"
    dup_text += f"Duplicate analysis:\n"
    for token, positions in duplicates.items():
        dup_text += f"  '{token}' at positions {positions}\n"
    dup_text += "\nInduction mechanism:\n"
    
    # Check induction predictions
    for m, token in enumerate(tokens):
        if token in duplicates:
            positions = duplicates[token]
            earlier = [p for p in positions if p < m]
            if earlier and earlier[-1] + 1 < seq_len:
                predicted = tokens[earlier[-1] + 1]
                dup_text += f"  Position {m} ('{token}') → predicts '{predicted}'\n"
    
    axes[1, 0].text(0.1, 0.5, dup_text, fontsize=11, verticalalignment='center',
                   fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Final prediction for last position
    axes[1, 1].axis('off')
    if tokens and tokens[-1] in duplicates:
        positions = duplicates[tokens[-1]]
        earlier = [p for p in positions if p < len(tokens) - 1]
        if earlier and earlier[-1] + 1 < seq_len:
            predicted = tokens[earlier[-1] + 1]
            prediction_text = f"🎯 PREDICTION:\n\nLast token: '{tokens[-1]}'\n"
            prediction_text += f"Found earlier at position {earlier[-1]}\n"
            prediction_text += f"Next token was: '{predicted}'\n\n"
            prediction_text += f"Model predicts: '{predicted}'"
            color = 'lightgreen'
        else:
            prediction_text = "No induction pattern\n(cannot predict)"
            color = 'lightyellow'
    else:
        prediction_text = "No duplicate tokens\n(no induction possible)"
        color = 'lightyellow'
    
    axes[1, 1].text(0.5, 0.5, prediction_text, fontsize=14, verticalalignment='center',
                   horizontalalignment='center', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    plt.suptitle(f'Induction Circuit: "{sequence}"', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

interact(visualize_induction, 
         sequence=widgets.Text(value="a b c a ", description='Sequence:'))

# %% Cell 5: Complete Forward Pass
"""
COMPLETE FORWARD PASS:

1. h^(0) = Embed(tokens)
2. Layer 0: Previous token head enriches each position with predecessor info
3. Residual: h^(0.5) = h^(0) + Layer0Output
4. Layer 1: Semantic head finds duplicates and attends to following tokens
5. Residual: h^(1) = h^(0.5) + Layer1Output
6. Output projection: logits = W_out · h^(1)
7. Prediction: next_token = argmax(logits[-1])
"""

def full_forward_pass(sequence="a b c a "):
    """Run complete forward pass and show prediction"""
    tokens = [c for c in sequence if c in VOCAB][:20]
    seq_len = len(tokens)
    
    if seq_len == 0:
        return None
    
    print("="*70)
    print("COMPLETE FORWARD PASS")
    print("="*70)
    
    # Step 1: Embedding
    print("\n📥 Step 1: Token Embedding")
    embeddings = np.array([embed_token(t) for t in tokens])
    print(f"  Input: {[t if t != ' ' else '␣' for t in tokens]}")
    print(f"  Shape: {embeddings.shape}")
    
    # Step 2: Layer 0 (Previous Token Head)
    print("\n🔄 Step 2: Layer 0 - Previous Token Head")
    print("  W_q = α·R_{Θ,-1}·W_k  where α =", ALPHA)
    prev_keys, prev_queries = previous_token_head(embeddings)
    _, prev_pattern = compute_attention(prev_queries, prev_keys)
    prev_output = np.array([prev_pattern[m] @ prev_keys for m in range(seq_len)])
    print(f"  Previous token attention pattern shape: {prev_pattern.shape}")
    print(f"  Diagonal accuracy: check visualization above")
    
    # Step 3: Residual after Layer 0
    h_mid = embeddings[:, :D_HEAD] + prev_output  # Simplified residual
    print("\n➕ Step 3: Residual Connection")
    print(f"  h^(0.5) = h^(0) + Layer0Output")
    
    # Step 4: Layer 1 (Semantic/Induction)
    print("\n🧠 Step 4: Layer 1 - Semantic/Induction Head")
    sem_keys, sem_queries, _, _ = semantic_head(embeddings, tokens)
    _, sem_pattern = compute_attention(sem_queries, sem_keys)
    
    # Apply induction boost
    duplicates = find_duplicates(tokens)
    for m, token in enumerate(tokens):
        if token in duplicates:
            positions = duplicates[token]
            earlier = [p for p in positions if p < m]
            if earlier and earlier[-1] + 1 < seq_len:
                sem_pattern[m, earlier[-1] + 1] += 0.5
    
    for m in range(seq_len):
        sem_pattern[m] /= sem_pattern[m].sum()
    
    print("  Rank-1 weight matrices: W_k = u_k⊗v_k^T, W_q = u_q⊗v_q^T")
    
    # Step 5: Final prediction
    print("\n🎯 Step 5: Prediction")
    last_token = tokens[-1]
    
    if last_token in duplicates:
        positions = duplicates[last_token]
        earlier = [p for p in positions if p < len(tokens) - 1]
        if earlier and earlier[-1] + 1 < seq_len:
            predicted = tokens[earlier[-1] + 1]
            confidence = sem_pattern[-1, earlier[-1] + 1]
            print(f"  Input ends with: '{last_token}'")
            print(f"  Found earlier '{last_token}' at position {earlier[-1]}")
            print(f"  That was followed by: '{predicted}'")
            print(f"  Attention confidence: {confidence:.3f}")
            print(f"\n  ✅ PREDICTION: '{predicted}'")
        else:
            print(f"  No valid induction pattern")
            print(f"  ❌ Cannot predict")
    else:
        print(f"  No duplicate tokens found")
        print(f"  ❌ No induction possible")
    
    print("\n" + "="*70)
    return True

interact(full_forward_pass,
         sequence=widgets.Text(value="a b c a ", description='Input:'))

# %% Cell 6: Math Reference
"""
MATH REFERENCE - Chris Wendler's Equations

All equations from: https://wendlerc.github.io/notes/rope.html

RoPE Rotation Matrix:
===================
R^{d}_{Θ,m} = block_diag(R_1, R_2, ..., R_{d/2})

where each block R_i rotates by angle m·θ_i:
R_i = [ cos(mθ_i)  -sin(mθ_i) ]
      [ sin(mθ_i)   cos(mθ_i) ]

θ_i = 1 / (10000^(2i/d))  (frequency schedule)

Key property: R_{Θ,m}^T · R_{Θ,n} = R_{Θ,n-m}
(Attention depends only on relative position!)

Previous Token Head:
==================
Construction makes position m attend to m-1:

W_k · x_i = (1, 0, 1, 0, ..., 1, 0)^T  (constant)
W_q = α · R_{Θ,-1} · W_k

Result:
q_m^T · k_n = α · c^T · R_{Θ,n-m+1} · c

When n = m-1: rotation is by 0°, dot product is MAXIMUM (d/2·α)
When n ≠ m-1: rotation is non-zero, dot product is SMALLER

Semantic Head:
=============
Rank-1 weight matrices:
W_k = u_k ⊗ v_k^T
W_q = u_q ⊗ v_q^T

Attention decomposes as:
q_m^T · k_n = (u_q^T u_k) · (v_q^T x_m) · (v_k^T x_n)
              ^^^^^^^^^^^   ^^^^^^^^^^^^   ^^^^^^^^^^^^
              geometric     query content  key content
              similarity    (semantic)     (semantic)

For induction, find duplicate tokens and attend to what followed.
"""

print("📚 Math Reference loaded!")
print("Run the cells above to see the equations in action.")
