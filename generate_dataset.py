"""
generate_dataset.py
-------------------
Generates a realistic 12,000-row salary dataset modeled after
real Kaggle salary/HR datasets (e.g., DS Salaries, HR Analytics).
Incorporates India-aware salary bands while keeping features
universally applicable.
"""

import numpy as np
import pandas as pd
import random

np.random.seed(42)
random.seed(42)

N = 12000

# ── Categorical pools ──────────────────────────────────────────────────────────
job_roles = {
    "Software Engineer":        (6,  20),
    "Senior Software Engineer": (10, 30),
    "Data Analyst":             (5,  18),
    "Data Scientist":           (8,  25),
    "ML Engineer":              (9,  28),
    "Product Manager":          (10, 32),
    "Business Analyst":         (5,  18),
    "HR Manager":               (5,  16),
    "Finance Analyst":          (5,  17),
    "Marketing Manager":        (6,  18),
    "DevOps Engineer":          (7,  22),
    "Cloud Architect":          (12, 35),
    "Cybersecurity Analyst":    (8,  24),
    "UI/UX Designer":           (5,  16),
    "QA Engineer":              (4,  14),
    "Fresher / Junior Dev":     (2,   7),
    "Operations Manager":       (6,  18),
    "Sales Executive":          (3,  12),
    "Tech Lead":                (14, 38),
    "Engineering Manager":      (18, 50),
}

industries = ["IT / Software", "Finance & Banking", "E-Commerce",
              "Healthcare", "Manufacturing", "Consulting",
              "Education", "Government/PSU", "Media & Entertainment", "Startup"]

education_levels = {
    "High School":   0,
    "Diploma":       1,
    "Bachelor's":    2,
    "Master's":      3,
    "MBA":           3,
    "PhD":           4,
}

company_sizes = ["Small (<50)", "Mid (50-500)", "Large (500-5000)", "Enterprise (5000+)"]
company_types = ["MNC", "Indian Large Corp", "Mid-size", "Startup", "Government/PSU"]
locations      = ["Bangalore", "Mumbai", "Delhi NCR", "Hyderabad", "Pune",
                  "Chennai", "Kolkata", "Ahmedabad", "Tier-2 City", "Remote"]

location_multiplier = {
    "Bangalore": 1.20, "Mumbai": 1.18, "Delhi NCR": 1.15,
    "Hyderabad": 1.12, "Pune": 1.10, "Chennai": 1.08,
    "Kolkata": 0.95, "Ahmedabad": 0.93, "Tier-2 City": 0.85, "Remote": 1.05,
}
company_type_mult = {
    "MNC": 1.20, "Indian Large Corp": 1.05, "Mid-size": 0.95,
    "Startup": 1.10, "Government/PSU": 0.80,
}
size_mult = {
    "Small (<50)": 0.88, "Mid (50-500)": 0.97,
    "Large (500-5000)": 1.08, "Enterprise (5000+)": 1.18,
}

def education_score(edu):
    return education_levels.get(edu, 2)

rows = []

for _ in range(N):
    role = random.choice(list(job_roles.keys()))
    base_min, base_max = job_roles[role]

    experience = round(np.random.exponential(scale=5), 1)
    experience = min(experience, 32)

    edu = random.choice(list(education_levels.keys()))
    edu_score = education_score(edu)

    industry  = random.choice(industries)
    location  = random.choice(locations)
    comp_type = random.choice(company_types)
    comp_size = random.choice(company_sizes)

    performance_rating = round(np.random.normal(3.3, 0.7), 1)
    performance_rating = max(1.0, min(5.0, performance_rating))

    num_skills          = int(np.random.poisson(4) + 1)
    num_skills          = min(num_skills, 12)
    certifications      = int(np.random.poisson(1))
    certifications      = min(certifications, 6)
    projects_completed  = int(np.random.poisson(8) + experience * 1.5)
    years_at_company    = round(min(experience, np.random.exponential(3)), 1)
    promotions_received = int(experience // 4 + np.random.binomial(2, 0.3))
    overtime_hours_week = int(np.clip(np.random.normal(5, 4), 0, 25))
    distance_km         = int(np.clip(np.random.exponential(20), 0, 100))
    monthly_hours       = int(np.clip(np.random.normal(176, 20), 120, 240))

    # ── Salary formula (₹ LPA) ─────────────────────────────────────────────────
    base = base_min + (base_max - base_min) * (experience / 30)
    base += edu_score * 1.2
    base += performance_rating * 1.5
    base += num_skills * 0.4
    base += certifications * 0.8
    base += promotions_received * 0.6
    base *= location_multiplier.get(location, 1.0)
    base *= company_type_mult.get(comp_type, 1.0)
    base *= size_mult.get(comp_size, 1.0)

    if industry in ["IT / Software", "Finance & Banking", "E-Commerce"]:
        base *= 1.12
    elif industry in ["Healthcare", "Consulting"]:
        base *= 1.05
    elif industry in ["Government/PSU", "Education"]:
        base *= 0.88

    base += np.random.normal(0, base * 0.08)
    base = max(1.5, round(base, 2))

    increment_pct = round(
        np.clip(performance_rating * 2.5 + np.random.normal(2, 1.5), 0, 40), 1
    )
    promotion_prob = round(
        np.clip(
            (performance_rating / 5) * 0.4
            + (promotions_received / max(experience, 1)) * 0.2
            + (num_skills / 12) * 0.2
            + np.random.normal(0, 0.05),
            0, 1,
        ),
        2,
    )

    rows.append({
        "job_role":            role,
        "industry":            industry,
        "education_level":     edu,
        "location":            location,
        "company_type":        comp_type,
        "company_size":        comp_size,
        "years_experience":    experience,
        "years_at_company":    years_at_company,
        "performance_rating":  performance_rating,
        "num_skills":          num_skills,
        "certifications":      certifications,
        "projects_completed":  projects_completed,
        "promotions_received": promotions_received,
        "overtime_hours_week": overtime_hours_week,
        "distance_from_work":  distance_km,
        "monthly_hours":       monthly_hours,
        "salary_lpa":          base,
        "increment_pct":       increment_pct,
        "promotion_prob":      promotion_prob,
    })

df = pd.DataFrame(rows)
df.to_csv("salary_dataset.csv", index=False)
print(f"✅ Dataset saved: {df.shape[0]} rows × {df.shape[1]} columns")
print(df.describe())
