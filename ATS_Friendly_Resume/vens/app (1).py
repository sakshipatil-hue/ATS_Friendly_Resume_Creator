from flask import Flask, render_template, request, Response, jsonify
from flask_cors import CORS
import requests
import json
import io
from weasyprint import HTML, CSS
from datetime import datetime
import logging
import re
from collections import Counter
from urllib.parse import urlparse
from typing import List, Dict, Tuple
import html2text
from difflib import SequenceMatcher

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"  # Change to your desired model

# In-memory history storage (session-based)
resume_history = []

# ATS-Friendly Resume Template
ATS_RESUME_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{full_name} - Resume</title>
    <style>
        @page {{
            size: A4;
            margin: 0.5in;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: Arial, Calibri, sans-serif;
            font-size: 11pt;
            line-height: 1.4;
            color: #000;
            background: #fff;
        }}
        
        .container {{
            max-width: 8.5in;
            margin: 0 auto;
            padding: 0;
        }}
        
        /* Header Section - ATS Friendly */
        .header {{
            border-bottom: 2px solid #000;
            padding-bottom: 8pt;
            margin-bottom: 10pt;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 14pt;
            font-weight: bold;
            margin: 0 0 4pt 0;
        }}
        
        .contact-info {{
            font-size: 10pt;
            line-height: 1.3;
        }}
        
        .contact-info span {{
            margin: 0 12pt 0 0;
        }}
        
        /* Section Headings - ATS Friendly */
        .section-title {{
            font-size: 12pt;
            font-weight: bold;
            margin-top: 10pt;
            margin-bottom: 6pt;
            border-bottom: 1pt solid #000;
            padding-bottom: 4pt;
            text-transform: uppercase;
        }}
        
        /* Professional Summary */
        .summary {{
            font-size: 10pt;
            line-height: 1.5;
            margin-bottom: 8pt;
            text-align: justify;
        }}
        
        /* Work Experience */
        .job {{
            margin-bottom: 8pt;
            page-break-inside: avoid;
        }}
        
        .job-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 2pt;
        }}
        
        .job-title {{
            font-weight: bold;
            font-size: 11pt;
        }}
        
        .job-date {{
            font-size: 10pt;
            text-align: right;
        }}
        
        .company {{
            font-style: italic;
            font-size: 10pt;
            margin-bottom: 3pt;
        }}
        
        .job-description {{
            margin-left: 0;
            font-size: 10pt;
            line-height: 1.4;
        }}
        
        .job-description li {{
            margin-left: 20pt;
            margin-bottom: 2pt;
        }}
        
        /* Education */
        .education-item {{
            margin-bottom: 6pt;
        }}
        
        .degree {{
            font-weight: bold;
            font-size: 11pt;
        }}
        
        .school {{
            font-style: italic;
            font-size: 10pt;
            margin: 2pt 0;
        }}
        
        /* Skills */
        .skills {{
            font-size: 10pt;
        }}
        
        .skill-category {{
            margin-bottom: 4pt;
        }}
        
        .skill-category-name {{
            font-weight: bold;
            display: inline;
        }}
        
        .skill-category-content {{
            display: inline;
        }}
        
        /* Certifications */
        .certification {{
            margin-bottom: 4pt;
            font-size: 10pt;
        }}
        
        /* No fancy formatting for ATS */
        em, i {{
            font-style: italic;
        }}
        
        strong, b {{
            font-weight: bold;
        }}
        
        /* Avoid tables - use simple layout */
        ul {{
            list-style-type: disc;
            margin-left: 20pt;
        }}
        
        ol {{
            list-style-type: decimal;
            margin-left: 20pt;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>{full_name}</h1>
            <div class="contact-info">
                <span>{email}</span>
                <span>{phone}</span>
                <span>{location}</span>
            </div>
        </div>
        
        <!-- Professional Summary -->
        {summary_section}
        
        <!-- Work Experience -->
        {experience_section}
        
        <!-- Education -->
        {education_section}
        
        <!-- Skills -->
        {skills_section}
        
        <!-- Certifications -->
        {certifications_section}
    </div>
</body>
</html>
"""

def extract_visible_text(html_content: str) -> str:
    """Extract visible text from HTML without JS execution"""
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.ignore_tables = False
    h.ignore_emphasis = False
    h.body_width = 0
    return h.handle(html_content)

def clean_and_tokenize(text: str) -> List[str]:
    """Clean text and tokenize into words"""
    # Remove HTML tags, URLs, and special characters
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s\-]', ' ', text.lower())
    
    # Tokenize
    tokens = re.findall(r'\b[a-z]{3,}\b', text)
    
    # Remove common stopwords
    stopwords = set([
        'the', 'and', 'for', 'with', 'this', 'that', 'have', 'from',
        'will', 'your', 'are', 'you', 'not', 'but', 'was', 'were',
        'has', 'been', 'they', 'their', 'what', 'which', 'who',
        'them', 'there', 'our', 'can', 'should', 'would', 'could'
    ])
    
    return [t for t in tokens if t not in stopwords]

def extract_top_keywords(text: str, top_n: int = 20) -> List[str]:
    """Extract top keywords by frequency"""
    tokens = clean_and_tokenize(text)
    if not tokens:
        return []
    
    # Count frequencies
    word_counts = Counter(tokens)
    
    # Get top N keywords (excluding very common words)
    common_job_words = {'job', 'work', 'experience', 'skills', 'required',
                       'ability', 'team', 'company', 'role', 'position'}
    
    top_keywords = []
    for word, count in word_counts.most_common(top_n * 2):
        if word not in common_job_words and len(word) > 2:
            top_keywords.append(word)
            if len(top_keywords) >= top_n:
                break
    
    return top_keywords

def predict_job_role(keywords: List[str]) -> str:
    """Simple rule-based job role prediction"""
    role_mappings = {
        'software': ['developer', 'engineer', 'programmer', 'python', 'java', 
                    'javascript', 'c++', 'react', 'node', 'backend', 'frontend'],
        'data': ['analyst', 'scientist', 'engineer', 'sql', 'python', 'analysis',
                'database', 'bigdata', 'warehouse', 'mining'],
        'devops': ['engineer', 'aws', 'azure', 'cloud', 'docker', 'kubernetes',
                  'ci/cd', 'terraform', 'infrastructure', 'deployment'],
        'designer': ['ui', 'ux', 'design', 'figma', 'sketch', 'prototype',
                    'wireframe', 'user', 'interface', 'experience'],
        'manager': ['project', 'product', 'team', 'lead', 'scrum', 'agile',
                   'kanban', 'budget', 'strategy', 'planning'],
        'analyst': ['business', 'financial', 'data', 'systems', 'requirements',
                   'process', 'analysis', 'modeling', 'reporting'],
        'engineer': ['mechanical', 'civil', 'electrical', 'system', 'network',
                    'security', 'quality', 'test', 'validation']
    }
    
    keyword_set = set(keywords)
    scores = {}
    
    for role, trigger_words in role_mappings.items():
        score = sum(1 for word in trigger_words if word in keyword_set)
        if score > 0:
            scores[role] = score
    
    if not scores:
        return "Professional"
    
    # Return role with highest score
    predicted = max(scores.items(), key=lambda x: x[1])[0]
    return predicted.title() + (" Engineer" if predicted in ['software', 'devops', 'data'] else 
                               " Analyst" if predicted == 'analyst' else 
                               " Manager" if predicted == 'manager' else "")

def calculate_ats_match(resume_skills: Dict, job_keywords: List[str]) -> Dict:
    """Calculate ATS keyword match percentage"""
    if not job_keywords:
        return {
            "ats_match_percent": 0,
            "matched_keywords": [],
            "missing_keywords": []
        }
    
    # Flatten resume skills
    all_resume_skills = []
    if isinstance(resume_skills, dict):
        for category, skills in resume_skills.items():
            if isinstance(skills, list):
                all_resume_skills.extend([s.lower() for s in skills])
    elif isinstance(resume_skills, list):
        all_resume_skills = [s.lower() for s in resume_skills]
    
    resume_skill_set = set(all_resume_skills)
    job_keyword_set = set([k.lower() for k in job_keywords])
    
    # Find matches (partial matches count)
    matched = []
    for job_word in job_keyword_set:
        # Check for exact match or partial match in resume skills
        for resume_skill in resume_skill_set:
            if job_word in resume_skill or resume_skill in job_word:
                matched.append(job_word)
                break
    
    matched_set = set(matched)
    missing = list(job_keyword_set - matched_set)[:10]  # Top 10 only
    
    match_percent = (len(matched_set) / len(job_keyword_set) * 100) if job_keyword_set else 0
    
    return {
        "ats_match_percent": round(match_percent, 1),
        "matched_keywords": list(matched_set),
        "missing_keywords": missing
    }

def calculate_resume_score(resume_data: Dict) -> Dict:
    """Calculate resume quality score with explanations"""
    score = 0
    max_score = 100
    explanations = []
    
    # Summary check (15 points)
    if resume_data.get('summary') and len(resume_data['summary'].strip()) > 50:
        score += 15
        explanations.append("✓ Professional summary present and detailed (+15)")
    else:
        explanations.append("✗ Professional summary missing or too brief (0/15)")
    
    # Experience check (30 points)
    experience = resume_data.get('experience', [])
    if experience and len(experience) > 0:
        score += min(30, len(experience) * 10)  # 10 points per job, max 30
        exp_count = len(experience)
        explanations.append(f"✓ {exp_count} work experience entries (+{min(30, exp_count * 10)})")
    else:
        explanations.append("✗ No work experience listed (0/30)")
    
    # Education check (20 points)
    education = resume_data.get('education', [])
    if education and len(education) > 0:
        score += 20
        explanations.append("✓ Education section complete (+20)")
    else:
        explanations.append("✗ Education section missing (0/20)")
    
    # Skills check (20 points)
    skills = resume_data.get('skills', {})
    if skills and len(skills) > 0:
        total_skills = sum(len(s) for s in skills.values() if isinstance(s, list))
        if total_skills >= 5:
            score += 20
            explanations.append(f"✓ {total_skills} skills listed across categories (+20)")
        else:
            score += 10
            explanations.append(f"⚠ Only {total_skills} skills listed (+10)")
    else:
        explanations.append("✗ Skills section missing (0/20)")
    
    # Certifications check (15 points)
    certs = resume_data.get('certifications', [])
    if certs and len(certs) > 0:
        score += min(15, len(certs) * 5)  # 5 points per cert, max 15
        explanations.append(f"✓ {len(certs)} certifications listed (+{min(15, len(certs) * 5)})")
    else:
        explanations.append("✗ No certifications listed (0/15)")
    
    return {
        "resume_score": score,
        "score_explanation": explanations
    }

# ===== NEW FEATURE 1: Resume-Job Alignment Diagnosis =====
def calculate_alignment_issues(resume_data: Dict, job_keywords: List[str], predicted_job_role: str, job_description: str) -> List[str]:
    """Rule-based resume-job alignment diagnosis"""
    alignment_issues = []
    
    # Extract role from resume (from summary and experience)
    resume_text = ""
    if resume_data.get('summary'):
        resume_text += resume_data['summary'].lower() + " "
    for exp in resume_data.get('experience', []):
        if exp.get('title'):
            resume_text += exp['title'].lower() + " "
    
    # Check if predicted role matches resume content
    resume_role_keywords = {
        'engineer': ['engineer', 'developer', 'architect', 'sre'],
        'analyst': ['analyst', 'consultant', 'advisor'],
        'manager': ['manager', 'director', 'head', 'lead'],
        'designer': ['designer', 'ux', 'ui', 'creative'],
        'scientist': ['scientist', 'researcher', 'ml', 'ai']
    }
    
    # Detect role mismatch
    if predicted_job_role != "Professional":
        predicted_lower = predicted_job_role.lower()
        resume_lower = resume_text.lower()
        
        # Check for seniority mismatch
        seniority_indicators = ['senior', 'lead', 'principal', 'manager', 'director', 'head']
        if any(word in predicted_lower for word in ['junior', 'entry', 'intern', 'associate']):
            if any(word in resume_lower for word in seniority_indicators):
                alignment_issues.append("⚠ Resume shows senior-level experience but JD targets junior/entry role")
        
        # Check for role focus mismatch
        if 'data' in predicted_lower and 'software' in resume_lower:
            alignment_issues.append("⚠ Resume emphasizes software engineering but JD targets data role")
        elif 'software' in predicted_lower and 'data' in resume_lower:
            alignment_issues.append("⚠ Resume emphasizes data work but JD targets software role")
    
    # Check summary-JD keyword alignment
    if job_description and resume_data.get('summary'):
        summary_lower = resume_data['summary'].lower()
        job_keywords_lower = [k.lower() for k in job_keywords]
        
        # Count keyword matches in summary
        summary_matches = sum(1 for kw in job_keywords_lower if kw in summary_lower)
        if len(job_keywords_lower) > 0:
            summary_match_percent = (summary_matches / len(job_keywords_lower)) * 100
            if summary_match_percent < 30:
                alignment_issues.append(f"⚠ Only {int(summary_match_percent)}% of JD keywords found in professional summary")
    
    return alignment_issues[:5]  # Return top 5 issues

# ===== NEW FEATURE 2: Section-wise ATS Contribution Score =====
def calculate_section_contribution(resume_data: Dict, job_keywords: List[str]) -> Dict:
    """Calculate ATS match contribution by section"""
    if not job_keywords:
        return {
            "summary": 25,
            "skills": 35,
            "experience": 25,
            "education": 10,
            "certifications": 5
        }
    
    job_keywords_lower = [k.lower() for k in job_keywords]
    contributions = {}
    total_weight = 0
    
    # Summary contribution (20% max)
    if resume_data.get('summary'):
        summary_text = resume_data['summary'].lower()
        summary_matches = sum(1 for kw in job_keywords_lower if kw in summary_text)
        contributions['summary'] = min(20, (summary_matches / len(job_keywords_lower)) * 20)
    else:
        contributions['summary'] = 0
    
    # Skills contribution (35% max)
    skills_text = ""
    if resume_data.get('skills'):
        for category, skills in resume_data['skills'].items():
            if isinstance(skills, list):
                skills_text += " ".join([s.lower() for s in skills]) + " "
    
    skills_matches = sum(1 for kw in job_keywords_lower if kw in skills_text)
    contributions['skills'] = min(35, (skills_matches / len(job_keywords_lower)) * 35)
    
    # Experience contribution (30% max)
    experience_text = ""
    for exp in resume_data.get('experience', []):
        if exp.get('title'):
            experience_text += exp['title'].lower() + " "
        if exp.get('descriptions'):
            for desc in exp['descriptions']:
                experience_text += desc.lower() + " "
    
    experience_matches = sum(1 for kw in job_keywords_lower if kw in experience_text)
    contributions['experience'] = min(30, (experience_matches / len(job_keywords_lower)) * 30)
    
    # Education contribution (10% max)
    education_text = ""
    for edu in resume_data.get('education', []):
        if edu.get('degree'):
            education_text += edu['degree'].lower() + " "
        if edu.get('school'):
            education_text += edu['school'].lower() + " "
    
    education_matches = sum(1 for kw in job_keywords_lower if kw in education_text)
    contributions['education'] = min(10, (education_matches / len(job_keywords_lower)) * 10)
    
    # Certifications contribution (5% max)
    certs_text = " ".join([c.lower() for c in resume_data.get('certifications', [])])
    certs_matches = sum(1 for kw in job_keywords_lower if kw in certs_text)
    contributions['certifications'] = min(5, (certs_matches / len(job_keywords_lower)) * 5)
    
    # Normalize to sum to 100
    total = sum(contributions.values())
    if total > 0:
        for key in contributions:
            contributions[key] = round((contributions[key] / total) * 100)
    
    # Ensure sum is exactly 100
    current_sum = sum(contributions.values())
    if current_sum != 100:
        # Adjust the largest component
        max_key = max(contributions, key=contributions.get)
        contributions[max_key] += (100 - current_sum)
    
    return contributions

# ===== NEW FEATURE 3: Smart Resume Rewrite Suggestions =====
def generate_rewrite_suggestions(resume_data: Dict, job_keywords: List[str], missing_keywords: List[str], predicted_job_role: str) -> List[str]:
    """Generate human-readable rewrite suggestions"""
    suggestions = []
    
    # Role-specific terminology suggestions
    role_terminology = {
        'software': ['developer', 'engineer', 'programmer', 'coding', 'development'],
        'data': ['analyst', 'scientist', 'engineer', 'analytics', 'processing'],
        'devops': ['engineer', 'operations', 'infrastructure', 'deployment', 'ci/cd'],
        'designer': ['designer', 'ux', 'ui', 'interface', 'experience', 'prototype'],
        'manager': ['manager', 'lead', 'director', 'oversaw', 'managed', 'led']
    }
    
    # Check summary for role-specific terminology
    if predicted_job_role != "Professional" and resume_data.get('summary'):
        summary_lower = resume_data['summary'].lower()
        predicted_lower = predicted_job_role.lower()
        
        for role, terms in role_terminology.items():
            if role in predicted_lower:
                missing_terms = [term for term in terms if term not in summary_lower]
                if missing_terms:
                    suggestions.append(f"Consider adding role-specific terms like '{missing_terms[0]}' to your professional summary")
                break
    
    # Missing keyword suggestions
    if missing_keywords:
        top_missing = missing_keywords[:3]
        suggestions.append(f"Add these missing keywords if relevant: {', '.join(top_missing)}")
    
    # Action verb suggestions
    action_verbs = ['developed', 'implemented', 'led', 'optimized', 'increased', 'reduced', 'managed', 'created']
    summary_text = resume_data.get('summary', '').lower()
    
    missing_verbs = [verb for verb in action_verbs if verb not in summary_text]
    if missing_verbs:
        suggestions.append(f"Strengthen summary with action verbs like '{missing_verbs[0]}'")
    
    # Quantification suggestions
    if resume_data.get('experience'):
        exp_text = str(resume_data['experience']).lower()
        if not any(word in exp_text for word in ['%', 'percent', 'increase', 'reduce', 'improve']):
            suggestions.append("Add quantifiable achievements (e.g., 'increased efficiency by 20%') to experience section")
    
    return suggestions[:5]  # Return top 5 suggestions

# ===== NEW FEATURE 4: Learning Roadmap Generator =====
def generate_learning_roadmap(missing_keywords: List[str]) -> List[str]:
    """Generate skill learning roadmap based on missing keywords"""
    if not missing_keywords:
        return [
            "Week 1-2: Review core skills in your target domain",
            "Week 3-4: Practice with real-world projects",
            "Week 5-6: Prepare for interviews with mock questions"
        ]
    
    roadmap = []
    tech_keywords = [kw for kw in missing_keywords if any(tech in kw for tech in [
        'python', 'java', 'javascript', 'react', 'node', 'sql', 'aws', 'docker',
        'machine', 'learning', 'data', 'analysis', 'cloud', 'devops'
    ])]
    
    if tech_keywords:
        roadmap.append(f"Week 1-2: Focus on mastering {tech_keywords[0]} fundamentals")
        if len(tech_keywords) > 1:
            roadmap.append(f"Week 3-4: Learn {tech_keywords[1]} and practical applications")
        roadmap.append(f"Week 5-6: Build a project combining {', '.join(tech_keywords[:2])}")
    else:
        soft_keywords = [kw for kw in missing_keywords if any(soft in kw for soft in [
            'leadership', 'communication', 'management', 'agile', 'scrum', 'team'
        ])]
        if soft_keywords:
            roadmap.append(f"Week 1-2: Develop {soft_keywords[0]} skills through workshops")
            roadmap.append(f"Week 3-4: Practice {soft_keywords[0]} in team settings")
            roadmap.append(f"Week 5-6: Document examples of {soft_keywords[0]} in action")
        else:
            roadmap = [
                "Week 1-2: Research industry trends and required skills",
                "Week 3-4: Identify and practice top 3 missing skills",
                "Week 5-6: Create portfolio projects showcasing new skills"
            ]
    
    return roadmap

# ===== NEW FEATURE 5: ATS Bias & Role-Level Warnings =====
def generate_ats_warnings(resume_data: Dict, predicted_job_role: str, job_description: str) -> List[str]:
    """Generate ATS bias and role-level warnings"""
    warnings = []
    
    # Extract text for analysis
    resume_text = ""
    if resume_data.get('summary'):
        resume_text += resume_data['summary'].lower()
    
    job_lower = job_description.lower() if job_description else ""
    predicted_lower = predicted_job_role.lower()
    
    # Seniority warnings
    senior_words = ['senior', 'lead', 'principal', 'manager', 'director', 'head', 'chief']
    junior_words = ['junior', 'entry', 'intern', 'associate', 'trainee', 'graduate']
    
    has_senior = any(word in resume_text for word in senior_words)
    targets_junior = any(word in predicted_lower or word in job_lower for word in junior_words)
    
    if has_senior and targets_junior:
        warnings.append("⚠ Overqualification warning: Resume shows senior experience but targets entry-level role")
    
    # Role mismatch warnings
    if predicted_job_role != "Professional":
        if 'intern' in predicted_lower and 'years' in resume_text:
            warnings.append("⚠ Experience mismatch: Internship roles typically expect less experience")
        
        if 'manager' in predicted_lower and not any(word in resume_text for word in ['manage', 'lead', 'team', 'oversee']):
            warnings.append("⚠ Missing management terminology for managerial role")
    
    # ATS format warnings (from existing resume data)
    if resume_data.get('skills'):
        skills_count = sum(len(s) for s in resume_data['skills'].values() if isinstance(s, list))
        if skills_count > 20:
            warnings.append("⚠ Skills section may be too long for ATS parsing")
    
    return warnings[:3]  # Return top 3 warnings

# ===== NEW FEATURE 6 & 7: History Management & Progress Comparison =====
def update_resume_history(entry: Dict) -> None:
    """Update resume analysis history"""
    global resume_history
    
    # Keep only last 10 entries
    if len(resume_history) >= 10:
        resume_history.pop(0)
    
    resume_history.append({
        'timestamp': datetime.now().isoformat(),
        'resume_score': entry.get('resume_score', 0),
        'ats_match_percent': entry.get('ats_match_percent', 0),
        'predicted_job_role': entry.get('predicted_job_role', 'Professional'),
        'job_title': entry.get('predicted_job_role', '')  # Using predicted role as job title
    })

def get_resume_progress() -> Dict:
    """Compare latest vs previous resume attempt"""
    global resume_history
    
    if len(resume_history) < 2:
        return {
            "previous_score": None,
            "current_score": None,
            "change": "No previous data"
        }
    
    previous = resume_history[-2]
    current = resume_history[-1]
    
    score_change = current['resume_score'] - previous['resume_score']
    ats_change = current['ats_match_percent'] - previous['ats_match_percent']
    
    return {
        "previous_score": previous['resume_score'],
        "current_score": current['resume_score'],
        "score_change": f"{'+' if score_change >= 0 else ''}{score_change}",
        "previous_ats": previous['ats_match_percent'],
        "current_ats": current['ats_match_percent'],
        "ats_change": f"{'+' if ats_change >= 0 else ''}{ats_change}",
        "improvement": "✓ Improved" if score_change > 0 else "⚠ Needs work" if score_change < 0 else "↔ Stable"
    }

def stream_from_ollama(prompt):
    """Stream response from Ollama API"""
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": True,
            "temperature": 0.7,
        }
        
        response = requests.post(OLLAMA_API_URL, json=payload, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if 'response' in data:
                    yield data['response']
                if data.get('done'):
                    break
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama. Make sure it's running on http://localhost:11434")
        yield "Error: Cannot connect to Ollama API. Make sure Ollama is running on localhost:11434"
    except Exception as e:
        logger.error(f"Error streaming from Ollama: {str(e)}")
        yield f"Error: {str(e)}"

def build_resume_html(resume_data):
    """Build ATS-friendly HTML resume from generated data"""
    
    # Professional Summary Section
    summary_section = ""
    if resume_data.get('summary'):
        summary_section = f'''
        <div class="section-title">Professional Summary</div>
        <div class="summary">{resume_data['summary']}</div>
        '''
    
    # Work Experience Section
    experience_section = ""
    if resume_data.get('experience'):
        experience_section = '<div class="section-title">Work Experience</div>'
        for job in resume_data['experience']:
            experience_section += f'''
            <div class="job">
                <div class="job-header">
                    <div class="job-title">{job.get('title', '')}</div>
                    <div class="job-date">{job.get('dates', '')}</div>
                </div>
                <div class="company">{job.get('company', '')}</div>
                <ul class="job-description">
                    {chr(10).join(f'<li>{desc}</li>' for desc in job.get('descriptions', []))}
                </ul>
            </div>
            '''
    
    # Education Section
    education_section = ""
    if resume_data.get('education'):
        education_section = '<div class="section-title">Education</div>'
        for edu in resume_data['education']:
            education_section += f'''
            <div class="education-item">
                <div class="degree">{edu.get('degree', '')}</div>
                <div class="school">{edu.get('school', '')}</div>
                <div style="font-size: 10pt;">{edu.get('details', '')}</div>
            </div>
            '''
    
    # Skills Section
    skills_section = ""
    if resume_data.get('skills'):
        skills_section = '<div class="section-title">Skills</div><div class="skills">'
        for skill_cat, skills in resume_data['skills'].items():
            skills_section += f'''
            <div class="skill-category">
                <span class="skill-category-name">{skill_cat}:</span>
                <span class="skill-category-content"> {', '.join(skills)}</span>
            </div>
            '''
        skills_section += '</div>'
    
    # Certifications Section
    certifications_section = ""
    if resume_data.get('certifications'):
        certifications_section = '<div class="section-title">Certifications</div>'
        for cert in resume_data['certifications']:
            certifications_section += f'<div class="certification">{cert}</div>'
    
    # Build final HTML
    html = ATS_RESUME_TEMPLATE.format(
        full_name=resume_data.get('full_name', 'Your Name'),
        email=resume_data.get('email', 'email@example.com'),
        phone=resume_data.get('phone', '(123) 456-7890'),
        location=resume_data.get('location', 'City, State'),
        summary_section=summary_section,
        experience_section=experience_section,
        education_section=education_section,
        skills_section=skills_section,
        certifications_section=certifications_section
    )
    
    return html

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/generate-resume', methods=['POST'])
def generate_resume():
    """Generate resume content using Ollama with streaming"""
    try:
        data = request.json
        user_input = data.get('input', '')
        job_description = data.get('job_description', '')
        job_url = data.get('job_url', '')
        
        if not user_input:
            return jsonify({'error': 'No input provided'}), 400
        
        # JOB DESCRIPTION ANALYSIS PIPELINE
        job_expected_keywords = []
        predicted_job_role = "Professional"
        missing_keywords = []
        
        if job_url:
            try:
                # Scrape visible text from URL
                response = requests.get(job_url, timeout=10)
                response.raise_for_status()
                job_description = extract_visible_text(response.text)
                logger.info(f"Scraped {len(job_description)} chars from {job_url}")
            except Exception as e:
                logger.warning(f"Failed to scrape {job_url}: {str(e)}")
        
        if job_description:
            # Extract keywords WITHOUT using LLM
            job_expected_keywords = extract_top_keywords(job_description, top_n=20)
            
            # Predict job role WITHOUT using LLM
            predicted_job_role = predict_job_role(job_expected_keywords)
            
            logger.info(f"Predicted role: {predicted_job_role}, Keywords: {len(job_expected_keywords)}")
        
        # Create prompt for resume generation WITH optimization guidance
        prompt = f"""You are an expert resume writer specializing in ATS optimization. Generate a professional, ATS-friendly resume based on the following information:

{user_input}

{'='*60 if job_description else ''}
{'JOB ANALYSIS CONTEXT:' if job_description else ''}
{'SYSTEM-COMPUTED VALUES (Use these only):' if job_description else ''}
{f'Target Role: {predicted_job_role}' if predicted_job_role != 'Professional' else ''}
{f'Key Keywords to include: {", ".join(job_expected_keywords[:])}' if job_expected_keywords else ''}
{'='*60 if job_description else ''}

Generate the resume in the following JSON format:
{{
    "full_name": "Full Name",
    "email": "email@example.com",
    "phone": "(123) 456-7890",
    "location": "City, State",
    "summary": "Professional summary here (** in 250 to 350 words **)",
    "experience": [
        {{
            "title": "Job Title",
            "company": "Company Name",
            "dates": "Jan 2020 - Dec 2021",
            "descriptions": ["Achievement 1", "Achievement 2"]
        }}
    ],
    "education": [
        {{
            "degree": "Degree Name",
            "school": "School Name",
            "details": "Graduation details"
        }}
    ],
    "skills": {{
        "Technical Skills": ["Skill1", "Skill2"],
        "Professional Skills": ["Skill3", "Skill4"]
    }},
    "certifications": ["Certification 1"]
}}

CRITICAL INSTRUCTIONS:
1. Use ONLY the information provided in the user input
2. DO NOT invent experience or skills that are not present
3. DO NOT analyze, score, or extract anything - only generate content
4. {'Use these system-computed keywords naturally: ' + ', '.join(job_expected_keywords[:]) if job_expected_keywords else ''}
5. {'Align with this target role: ' + predicted_job_role if predicted_job_role != 'Professional' else ''}
6. Make it ATS-friendly: use standard fonts, avoid graphics/tables, use action verbs
7. Quantify achievements where possible
8. Output ONLY valid JSON"""

        def stream_response():
            """Stream the response from Ollama"""
            yield 'data: {"type": "start"}\n\n'
            
            full_response = ""
            for chunk in stream_from_ollama(prompt):
                full_response += chunk
                yield f'data: {json.dumps({"type": "chunk", "content": chunk})}\n\n'
            
            # Try to parse and validate JSON
            try:
                # Find JSON in response
                start_idx = full_response.find('{')
                end_idx = full_response.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = full_response[start_idx:end_idx]
                    resume_data = json.loads(json_str)
                    
                    # CALCULATE RESUME QUALITY SCORE
                    score_data = calculate_resume_score(resume_data)
                    
                    # CALCULATE ATS MATCH ANALYSIS
                    ats_data = calculate_ats_match(resume_data.get('skills', {}), job_expected_keywords)
                    
                    # ===== ADD NEW FEATURES =====
                    # 1. Alignment Issues
                    alignment_issues = calculate_alignment_issues(
                        resume_data, job_expected_keywords, predicted_job_role, job_description
                    )
                    
                    # 2. Section-wise ATS Contribution
                    ats_section_contribution = calculate_section_contribution(resume_data, job_expected_keywords)
                    
                    # 3. Rewrite Suggestions
                    rewrite_suggestions = generate_rewrite_suggestions(
                        resume_data, job_expected_keywords, ats_data['missing_keywords'], predicted_job_role
                    )
                    
                    # 4. Learning Roadmap
                    learning_roadmap = generate_learning_roadmap(ats_data['missing_keywords'])
                    
                    # 5. ATS Warnings
                    ats_warnings = generate_ats_warnings(resume_data, predicted_job_role, job_description)
                    
                    # Add optimization data to resume
                    resume_data['predicted_job_role'] = predicted_job_role
                    resume_data['job_expected_keywords'] = job_expected_keywords
                    resume_data['ats_match_percent'] = ats_data['ats_match_percent']
                    resume_data['resume_score'] = score_data['resume_score']
                    resume_data['score_explanation'] = score_data['score_explanation']
                    resume_data['missing_keywords'] = ats_data['missing_keywords']
                    
                    # Add new features data
                    resume_data['alignment_issues'] = alignment_issues
                    resume_data['ats_section_contribution'] = ats_section_contribution
                    resume_data['rewrite_suggestions'] = rewrite_suggestions
                    resume_data['learning_roadmap'] = learning_roadmap
                    resume_data['ats_warnings'] = ats_warnings
                    
                    # ===== HISTORY TRACKING =====
                    # Add to history
                    history_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'resume_score': resume_data['resume_score'],
                        'ats_match_percent': resume_data['ats_match_percent'],
                        'predicted_job_role': predicted_job_role,
                        'job_title': predicted_job_role
                    }
                    update_resume_history(history_entry)
                    
                    # Get progress comparison
                    resume_data['resume_progress'] = get_resume_progress()
                    resume_data['history_summary'] = {
                        'total_analyses': len(resume_history),
                        'average_score': round(sum(h['resume_score'] for h in resume_history) / len(resume_history)) if resume_history else 0,
                        'best_score': max(h['resume_score'] for h in resume_history) if resume_history else 0
                    }
                    
                    yield f'data: {json.dumps({"type": "complete", "data": resume_data})}\n\n'
                else:
                    raise ValueError("No valid JSON found in response")
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {str(e)}")
                yield f'data: {json.dumps({"type": "error", "message": "Failed to parse resume data"})}\n\n'
        
        return Response(stream_response(), mimetype='text/event-stream')
    
    except Exception as e:
        logger.error(f"Error generating resume: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-pdf', methods=['POST'])
def generate_pdf():
    """Generate PDF from resume data"""
    try:
        resume_data = request.json
        
        # Build HTML
        html_content = build_resume_html(resume_data)
        
        # Generate PDF
        pdf_bytes = HTML(string=html_content).write_pdf()
        
        # Return PDF
        response = Response(pdf_bytes, mimetype='application/pdf')
        response.headers['Content-Disposition'] = f'attachment; filename="resume_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf"'
        
        return response
    
    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'model': MODEL_NAME})

@app.route('/landing')
def landing():
    """Serve the landing page"""
    return render_template('landing.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)