"""
prepare_education_knowledge_simple.py
Prepares education-specific knowledge base for hallucination detection.
Focuses on educational content, curriculum, pedagogy, and academic facts.
Simplified version without external dependencies.
"""
import os
import json
from typing import List, Dict, Any
import re

# Education-specific topics and categories
EDUCATION_TOPICS = {
    "pedagogy": [
        "Constructivism (learning theory)", "Behaviorism", "Cognitivism", "Social learning theory",
        "Multiple intelligences", "Bloom's taxonomy", "Differentiated instruction", "Project-based learning",
        "Flipped classroom", "Blended learning", "Montessori method", "Waldorf education",
        "Reggio Emilia approach", "Inquiry-based learning", "Experiential learning"
    ],
    "curriculum": [
        "Common Core State Standards", "International Baccalaureate", "Advanced Placement",
        "STEM education", "STEAM education", "Liberal arts education", "Vocational education",
        "Special education", "Gifted education", "Early childhood education", "Adult education"
    ],
    "subjects": [
        "Mathematics education", "Science education", "Language arts", "History education",
        "Geography education", "Physical education", "Art education", "Music education",
        "Computer science education", "Economics education", "Psychology education"
    ],
    "educational_psychology": [
        "Learning styles", "Memory and learning", "Motivation in education", "Classroom management",
        "Assessment and evaluation", "Educational technology", "Inclusive education",
        "Learning disabilities", "Gifted and talented education", "Educational neuroscience"
    ],
    "educational_systems": [
        "Education in the United States", "Education in the United Kingdom", "Education in Finland",
        "Education in Singapore", "Education in Japan", "Education in Canada", "Education in Australia",
        "International education", "Higher education", "Distance education", "Online learning"
    ]
}

def get_curriculum_facts():
    """Generate curriculum and educational standard facts."""
    print("Generating curriculum facts...")
    facts = []
    
    # Common Core Standards facts
    common_core_facts = [
        "Common Core State Standards are educational standards for English language arts and mathematics for grades K-12.",
        "The Common Core standards were developed to ensure students graduate from high school prepared for college and careers.",
        "Common Core mathematics standards focus on conceptual understanding, procedural skills, and problem-solving.",
        "Common Core English language arts standards emphasize reading, writing, speaking, and listening skills.",
        "The Common Core standards were adopted by 41 states, the District of Columbia, and four territories.",
        "Common Core standards are not a curriculum but a set of learning goals that outline what students should know and be able to do.",
        "The Common Core State Standards Initiative was launched in 2009 by state leaders and education experts.",
        "Common Core standards are designed to be internationally benchmarked and aligned with college and work expectations."
    ]
    facts.extend(common_core_facts)
    
    # STEM education facts
    stem_facts = [
        "STEM education integrates Science, Technology, Engineering, and Mathematics in an interdisciplinary approach.",
        "STEM education emphasizes hands-on, inquiry-based learning and real-world problem solving.",
        "The goal of STEM education is to prepare students for careers in science, technology, engineering, and mathematics.",
        "STEM education promotes critical thinking, creativity, collaboration, and communication skills.",
        "Many countries have implemented STEM education initiatives to address workforce needs in technical fields.",
        "STEAM education adds Arts to STEM, emphasizing creativity and design thinking.",
        "STEM education often involves project-based learning and real-world applications.",
        "The term STEM was first coined by the National Science Foundation in the 1990s."
    ]
    facts.extend(stem_facts)
    
    # Assessment facts
    assessment_facts = [
        "Formative assessment is used during learning to provide feedback and guide instruction.",
        "Summative assessment evaluates student learning at the end of an instructional period.",
        "Authentic assessment measures students' ability to apply knowledge in real-world contexts.",
        "Rubrics provide clear criteria for evaluating student work and performance.",
        "Portfolio assessment collects samples of student work over time to demonstrate learning progress.",
        "Standardized testing is a form of assessment that uses consistent procedures for administration and scoring.",
        "Performance-based assessment evaluates students' ability to perform specific tasks or demonstrate skills.",
        "Peer assessment involves students evaluating each other's work according to established criteria."
    ]
    facts.extend(assessment_facts)
    
    # International education facts
    international_facts = [
        "The International Baccalaureate (IB) offers four educational programmes for students aged 3 to 19.",
        "IB programmes focus on developing inquiring, knowledgeable, and caring young people.",
        "Advanced Placement (AP) courses are college-level courses offered in high schools.",
        "AP exams are scored on a scale of 1 to 5, with 3 being considered passing.",
        "Finland's education system is often cited as one of the best in the world.",
        "Singapore's education system emphasizes mathematics and science education.",
        "Japan's education system places high value on academic achievement and discipline.",
        "The United Kingdom's education system includes primary, secondary, and higher education levels."
    ]
    facts.extend(international_facts)
    
    print(f"✔️ Curriculum facts: {len(facts)}")
    return facts

def get_pedagogical_facts():
    """Generate pedagogical and teaching methodology facts."""
    print("Generating pedagogical facts...")
    facts = []
    
    # Learning theories
    learning_theories = [
        "Constructivism posits that learners construct knowledge through experience and reflection.",
        "Behaviorism focuses on observable behaviors and external stimuli in learning.",
        "Cognitivism emphasizes mental processes like memory, thinking, and problem-solving in learning.",
        "Social learning theory suggests that people learn through observation, imitation, and modeling.",
        "Multiple intelligences theory proposes that intelligence is not a single ability but multiple abilities.",
        "Vygotsky's sociocultural theory emphasizes the role of social interaction in cognitive development.",
        "Piaget's theory of cognitive development describes how children construct knowledge through stages.",
        "Bandura's social learning theory emphasizes the importance of modeling and observational learning."
    ]
    facts.extend(learning_theories)
    
    # Teaching methods
    teaching_methods = [
        "Differentiated instruction adapts teaching to meet individual student needs and learning styles.",
        "Project-based learning engages students in complex, real-world projects to develop knowledge and skills.",
        "Flipped classroom reverses traditional learning by delivering instructional content outside of class.",
        "Blended learning combines face-to-face instruction with online learning activities.",
        "Inquiry-based learning encourages students to ask questions and investigate to find answers.",
        "Cooperative learning involves students working together in small groups to achieve common goals.",
        "Direct instruction is a teacher-centered approach that provides explicit, systematic instruction.",
        "Discovery learning allows students to explore and discover concepts on their own."
    ]
    facts.extend(teaching_methods)
    
    # Classroom management
    classroom_management = [
        "Positive reinforcement increases desired behaviors through rewards and recognition.",
        "Classroom rules should be clear, consistent, and developed with student input.",
        "Proximity control involves moving closer to students to prevent or address misbehavior.",
        "Time management in classrooms includes transitions, instructional time, and assessment time.",
        "Building positive relationships with students improves classroom behavior and learning outcomes.",
        "Classroom routines help establish predictable patterns and reduce disruptive behavior.",
        "Active supervision involves circulating around the classroom and monitoring student behavior.",
        "Conflict resolution strategies help students resolve disagreements peacefully."
    ]
    facts.extend(classroom_management)
    
    # Learning styles and individual differences
    learning_styles = [
        "Visual learners prefer to learn through images, diagrams, and written directions.",
        "Auditory learners learn best through listening and verbal communication.",
        "Kinesthetic learners prefer hands-on activities and physical movement.",
        "Reading/writing learners prefer to learn through written words and text.",
        "Learning disabilities are neurological differences that affect how individuals process information.",
        "Gifted and talented students often need differentiated instruction to meet their advanced needs.",
        "Individualized Education Programs (IEPs) are designed for students with special needs.",
        "Universal Design for Learning (UDL) provides multiple means of representation, expression, and engagement."
    ]
    facts.extend(learning_styles)
    
    print(f"✔️ Pedagogical facts: {len(facts)}")
    return facts

def get_subject_specific_facts():
    """Generate subject-specific educational facts."""
    print("Generating subject-specific facts...")
    facts = []
    
    # Mathematics education
    math_facts = [
        "Mathematics education focuses on developing mathematical thinking and problem-solving skills.",
        "The National Council of Teachers of Mathematics (NCTM) sets standards for mathematics education.",
        "Mathematical modeling helps students apply mathematics to real-world situations.",
        "Number sense is the ability to understand numbers and their relationships.",
        "Mathematical reasoning involves logical thinking and proof development.",
        "Geometry education helps students understand spatial relationships and properties of shapes.",
        "Algebra education introduces students to symbolic reasoning and equation solving.",
        "Statistics education teaches students to collect, analyze, and interpret data."
    ]
    facts.extend(math_facts)
    
    # Science education
    science_facts = [
        "Science education emphasizes inquiry-based learning and scientific method.",
        "The Next Generation Science Standards (NGSS) guide science education in the United States.",
        "Laboratory work is essential for developing scientific skills and understanding.",
        "Scientific literacy involves understanding scientific concepts and processes.",
        "Environmental education promotes understanding of ecological systems and sustainability.",
        "Technology integration in science education enhances learning experiences.",
        "Science education should include both content knowledge and scientific practices.",
        "Cross-cutting concepts in science connect different scientific disciplines."
    ]
    facts.extend(science_facts)
    
    # Language arts education
    language_facts = [
        "Language arts education includes reading, writing, speaking, and listening skills.",
        "Phonics instruction teaches the relationship between sounds and letters.",
        "Reading comprehension strategies help students understand and interpret text.",
        "Writing instruction includes narrative, informational, and argumentative writing.",
        "Vocabulary development is essential for reading comprehension and communication.",
        "Grammar instruction helps students understand language structure and conventions.",
        "Literature study exposes students to diverse perspectives and cultural experiences.",
        "Media literacy helps students critically analyze various forms of communication."
    ]
    facts.extend(language_facts)
    
    # Computer science education
    cs_facts = [
        "Computer science education teaches computational thinking and programming skills.",
        "Coding education helps students develop logical thinking and problem-solving abilities.",
        "Digital literacy is essential for navigating the modern information landscape.",
        "Cybersecurity education teaches students about online safety and digital citizenship.",
        "Artificial intelligence education introduces students to machine learning concepts.",
        "Data science education helps students understand data analysis and visualization.",
        "Computational thinking involves breaking down problems into manageable parts.",
        "Algorithm design teaches students to create step-by-step solutions to problems."
    ]
    facts.extend(cs_facts)
    
    print(f"✔️ Subject-specific facts: {len(facts)}")
    return facts

def get_educational_technology_facts():
    """Generate educational technology facts."""
    print("Generating educational technology facts...")
    facts = []
    
    # Educational technology
    edtech_facts = [
        "Educational technology enhances teaching and learning through digital tools and resources.",
        "Learning Management Systems (LMS) provide platforms for online course delivery.",
        "Digital assessment tools offer immediate feedback and adaptive testing capabilities.",
        "Virtual reality (VR) in education creates immersive learning experiences.",
        "Augmented reality (AR) overlays digital information on the real world.",
        "Mobile learning allows students to access educational content on portable devices.",
        "Gamification uses game elements to increase student engagement and motivation.",
        "Adaptive learning technology personalizes instruction based on student performance.",
        "Online collaboration tools enable students to work together remotely.",
        "Digital portfolios allow students to showcase their work and progress over time.",
        "Educational apps provide interactive learning experiences across various subjects.",
        "Video conferencing tools facilitate remote learning and virtual classrooms.",
        "Cloud computing enables access to educational resources from anywhere.",
        "Artificial intelligence in education can provide personalized tutoring and assessment.",
        "Blockchain technology can be used for secure credentialing and academic records."
    ]
    facts.extend(edtech_facts)
    
    print(f"✔️ Educational technology facts: {len(facts)}")
    return facts

def save_education_knowledge(facts: List[str], output_file: str = "data/education_knowledge.txt"):
    """Save education-specific facts to file."""
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Deduplicate and clean facts
    unique_facts = list(set(facts))
    cleaned_facts = []
    
    for fact in unique_facts:
        # Clean the fact
        cleaned = re.sub(r'\s+', ' ', fact.strip())
        if len(cleaned) > 10 and len(cleaned) < 1000:  # Reasonable length
            cleaned_facts.append(cleaned)
    
    print(f"Total unique education facts: {len(cleaned_facts)}")
    
    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        for fact in cleaned_facts:
            f.write(fact + "\n")
    
    print(f"✅ Education knowledge saved to {output_file}!")
    
    # Also save as JSON for structured access
    json_file = output_file.replace('.txt', '.json')
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({
            "education_facts": cleaned_facts,
            "total_facts": len(cleaned_facts),
            "categories": list(EDUCATION_TOPICS.keys()),
            "sources": ["curriculum_standards", "pedagogical_theories", "subject_specific", "educational_technology"]
        }, f, indent=2)
    
    print(f"✅ Education knowledge also saved as JSON to {json_file}!")
    
    return cleaned_facts

def main():
    """Main function to prepare education knowledge base."""
    print("🎓 Preparing Education-Specific Knowledge Base")
    print("=" * 50)
    
    all_facts = []
    
    # Get facts from different sources
    all_facts.extend(get_curriculum_facts())
    all_facts.extend(get_pedagogical_facts())
    all_facts.extend(get_subject_specific_facts())
    all_facts.extend(get_educational_technology_facts())
    
    # Save the knowledge base
    cleaned_facts = save_education_knowledge(all_facts)
    
    print("\n🎉 Education knowledge base preparation complete!")
    print(f"Total facts collected: {len(cleaned_facts)}")
    print("\n📊 Knowledge Base Statistics:")
    print(f"- Curriculum & Standards: {len([f for f in cleaned_facts if any(term in f.lower() for term in ['common core', 'stem', 'assessment', 'international'])])}")
    print(f"- Pedagogical Theories: {len([f for f in cleaned_facts if any(term in f.lower() for term in ['constructivism', 'behaviorism', 'cognitivism', 'learning theory'])])}")
    print(f"- Subject-Specific: {len([f for f in cleaned_facts if any(term in f.lower() for term in ['mathematics', 'science', 'language', 'computer'])])}")
    print(f"- Educational Technology: {len([f for f in cleaned_facts if any(term in f.lower() for term in ['technology', 'digital', 'online', 'virtual'])])}")

if __name__ == "__main__":
    main() 