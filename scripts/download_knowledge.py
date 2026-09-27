"""
Script to download a free OpenStax Physics PDF for the RAG knowledge base.
Uses a small chapter extract if the full PDF is too large.

Run:
    python scripts/download_knowledge.py

This downloads a sample mechanics content to data/knowledge/
for immediate use without requiring the user to source a PDF manually.
"""

import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
KNOWLEDGE_DIR = ROOT / "data" / "knowledge"


OPENAI_MECHANICS_TEXT = """
Newton's Laws of Motion

Newton's First Law — Law of Inertia
An object at rest remains at rest and an object in motion remains in motion at 
constant speed and in a straight line unless acted on by a net external force.
This is often called the law of inertia. The tendency of an object to remain in 
its state of rest or uniform motion is called inertia.

Newton's Second Law
The acceleration of an object is directly proportional to the net force applied 
on it and inversely proportional to its mass.
F = ma
where F is net force (Newtons), m is mass (kilograms), a is acceleration (m/s²).

Example: A 5 kg block experiences a net force of 20 N. 
Its acceleration = F/m = 20/5 = 4 m/s².

Newton's Third Law — Action and Reaction
For every action there is an equal and opposite reaction.
If object A exerts force on object B, then object B exerts an equal and opposite 
force on object A. These forces always act on different objects.

Gravitational Force
F = G × (m1 × m2) / r²
G = 6.674 × 10⁻¹¹ N·m²/kg² (universal gravitational constant)
m1, m2 = masses of the two objects
r = distance between their centers

Weight = mg where g = 9.8 m/s² (gravitational acceleration on Earth's surface)

Work and Energy

Work-Energy Theorem
The net work done on an object equals the change in its kinetic energy.
W_net = ΔKE = ½mv² - ½mv₀²

Work = Force × displacement × cos(θ)
W = F·d·cos(θ)
where θ is angle between force and displacement direction.

Kinetic Energy: KE = ½mv²
Potential Energy (gravitational): PE = mgh

Conservation of Energy
In an isolated system, the total mechanical energy (KE + PE) remains constant 
when only conservative forces act.
KE₁ + PE₁ = KE₂ + PE₂

Momentum

Linear Momentum: p = mv (mass × velocity), vector quantity

Law of Conservation of Momentum
The total momentum of a closed system remains constant when no external forces act.
p_initial = p_final
m₁v₁ + m₂v₂ = m₁v₁' + m₂v₂'

Impulse: J = F·Δt = Δp (impulse equals change in momentum)

Collision Types:
1. Elastic: both momentum and kinetic energy conserved
2. Inelastic: only momentum conserved, KE not conserved
3. Perfectly inelastic: objects stick together after collision

Projectile Motion

A projectile moves with constant horizontal velocity and constant vertical 
acceleration due to gravity.

Horizontal: x = v₀ₓ·t (constant velocity)
Vertical: y = v₀ᵧ·t - ½gt² (accelerated by gravity)

Range: R = v₀²·sin(2θ)/g
Maximum height: H = v₀²·sin²(θ)/(2g)
Time of flight: T = 2v₀·sin(θ)/g

where θ is launch angle, v₀ is initial speed.

Circular Motion

Centripetal acceleration: ac = v²/r (directed toward center)
Centripetal force: Fc = mv²/r

Period: T = 2πr/v
Frequency: f = 1/T

For a satellite in circular orbit:
GMm/r² = mv²/r
v = √(GM/r)

Friction

Static friction: fs ≤ μs·N (prevents motion, up to maximum value)
Kinetic friction: fk = μk·N (opposes sliding motion)

μs > μk always (static coefficient > kinetic coefficient)
N = Normal force perpendicular to surface.

Hooke's Law (Springs)
F = -kx
where k is spring constant (N/m) and x is displacement from equilibrium.
Elastic potential energy: PE_spring = ½kx²

Simple Harmonic Motion (SHM)
A restoring force proportional to displacement: F = -kx
Period of a spring-mass system: T = 2π√(m/k)
Period of a simple pendulum: T = 2π√(L/g)

Angular quantities:
ω = 2πf (angular frequency)
Position: x(t) = A·cos(ωt + φ)
A = amplitude, φ = phase angle

Torque and Rotational Motion

Torque: τ = r × F = rF·sin(θ)
Moment of inertia: I (rotational equivalent of mass)
Newton's second law for rotation: τ = I·α (α = angular acceleration)

Angular momentum: L = I·ω
Conservation of angular momentum: L_initial = L_final (when no external torque)

Scalars and Vectors

Scalar quantities have magnitude only: speed, mass, temperature, energy, time.
Vector quantities have both magnitude and direction: velocity, force, acceleration, displacement, momentum.

Vector addition: Add components separately.
If A = (Ax, Ay) and B = (Bx, By):
A + B = (Ax+Bx, Ay+By)

Magnitude: |A| = √(Ax² + Ay²)
Direction: θ = arctan(Ay/Ax)

Kinematic Equations (constant acceleration)

v = v₀ + at
x = x₀ + v₀t + ½at²
v² = v₀² + 2a(x - x₀)
x = x₀ + ½(v₀ + v)t

where: v₀ = initial velocity, v = final velocity, a = acceleration, t = time, x = position.
"""

def create_sample_text():
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = KNOWLEDGE_DIR / "mechanics.txt"
    
    if output_path.exists():
        print(f"Already exists: {output_path}")
        return
    
    output_path.write_text(OPENAI_MECHANICS_TEXT, encoding="utf-8")
    print(f"Created: {output_path} ({len(OPENAI_MECHANICS_TEXT)} chars)")
    print("\nNote: This is a text file. For better RAG quality, replace with:")
    print("  - OpenStax University Physics Vol 1 (free PDF at openstax.org)")
    print("  - NCERT Physics Class 11 (free at ncert.nic.in)")


if __name__ == "__main__":
    create_sample_text()
    print("\nNext step: python -m rag.ingestion")
