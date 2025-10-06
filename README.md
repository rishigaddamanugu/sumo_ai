## 🤖 Sumo AI — Competitive Reinforcement Learning in Unity  

Sumo AI is a **Unity-based multi-agent reinforcement learning project** that explores competitive behavior through **self-play training**. Two agents battle in a circular sumo arena, learning to push each other out while maintaining balance and adapting to evolving opponent strategies.  

Unlike typical Unity ML-Agent setups, this project features a **custom Proximal Policy Optimization (PPO)** implementation built in Python. The trainer communicates with Unity in real time through **WebSockets**, allowing synchronized physics simulation, reward exchange, and model updates.  

The experiment demonstrates how emergent behaviors arise naturally from simple objectives in self-play settings, showing that complex strategies can evolve without explicit instruction.  

---

### 🎯 Objectives  
- Investigate **emergent competition and adaptation** through self-play reinforcement learning.  
- Replace Unity ML-Agents with a fully **custom PPO backend** implemented in Python.  
- Study the stability and convergence of shared-policy training across thousands of episodes.  

---

### ⚙️ Key Features  

#### 🧩 Multi-Agent Environment  
- Two Unity-controlled agents trained under a **shared PPO policy**.  
- Agents receive continuous state observations and rewards through a lightweight WebSocket API.  
- Includes dynamic physics interactions, balance mechanics, and positional awareness.  

#### 🔁 Custom PPO Implementation  
- Implemented a **from-scratch PPO algorithm** in Python for policy and value network optimization.  
- Communicates with Unity step-by-step to update actions, collect rewards, and compute gradients.  
- Supports both synchronous and asynchronous training modes for scalability.  

#### 🧠 Emergent Strategy Formation  
- Agents learn advanced behaviors such as **edge defense**, **momentum exploitation**, and **timing-based attacks**.  
- Demonstrates the spontaneous evolution of tactics without hard-coded logic.  
- Highlights the power of self-play reinforcement learning in continuous control tasks.  

#### 📈 Training Scale  
- Over **500,000 self-play episodes** run to achieve consistent performance and stability.  
- Evaluated convergence, policy variance, and win-rate across multiple seeds.  
- Uses custom logging and visualization tools for performance tracking.  

---

### 🧩 Tech Stack  
Python • Unity • WebSockets • PyTorch • Reinforcement Learning • PPO  

---

### 🚀 Vision  
Sumo AI aims to serve as an open platform for experimenting with **adversarial multi-agent learning** in physically realistic environments. It demonstrates that complex competitive intelligence can arise from simple reward structures when agents learn together under shared dynamics.
