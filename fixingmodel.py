import nltk
# import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier
import random
from langdetect import detect

# Kamus jawaban
answers = {
     "company": {
        "en": [
            "Our company was founded in 2005 with a focus on innovation and customer satisfaction.",
            "We have been in business for over 15 years, providing quality products and services.",
            "What sets us apart is our commitment to excellence and our dedicated team.",
            "Our mission is to provide the best possible solutions for our customers.",
            "The culture at our company is one of collaboration, innovation, and continuous improvement."
        ],
        "id": [
            "Perusahaan kami didirikan pada tahun 2005 dengan fokus pada inovasi dan kepuasan pelanggan.",
            "Kami telah beroperasi selama lebih dari 15 tahun, menyediakan produk dan layanan berkualitas.",
            "Yang membedakan kami adalah komitmen kami terhadap keunggulan dan tim yang berdedikasi.",
            "Misi kami adalah untuk memberikan solusi terbaik bagi pelanggan kami.",
            "Budaya di perusahaan kami adalah kerjasama, inovasi, dan perbaikan terus-menerus."
        ]
    },
    "team": {
        "en": [
            "Our team consists of experts in their respective fields, with years of experience in the industry.",
            "We have a diverse team that brings unique perspectives to our projects.",
            "Collaboration is key in our team, and we often hold brainstorming sessions to generate new ideas.",
            "We offer regular training sessions and opportunities for professional development to our team members."
        ],
        "id": [
            "Tim kami terdiri dari ahli di bidang masing-masing, dengan pengalaman bertahun-tahun dalam industri.",
            "Kami memiliki tim yang beragam yang membawa perspektif unik ke proyek-proyek kami.",
            "Kerjasama adalah kunci di tim kami, dan kami sering mengadakan sesi brainstorming untuk menghasilkan ide-ide baru.",
            "Kami menawarkan sesi pelatihan reguler dan kesempatan untuk pengembangan profesional kepada anggota tim kami."
        ]
    },
    "values": {
        "en": [
            "Our core values include integrity, innovation, and customer focus.",
            "We ensure our values are reflected in our work through regular reviews and feedback.",
            "We are actively involved in social and environmental initiatives, including charity work and sustainability programs.",
            "Diversity and inclusion are priorities in our company, and we strive to create a welcoming and inclusive environment for all."
        ],
        "id": [
            "Nilai inti kami termasuk integritas, inovasi, dan fokus pada pelanggan.",
            "Kami memastikan nilai-nilai kami tercermin dalam pekerjaan kami melalui tinjauan dan umpan balik secara berkala.",
            "Kami aktif terlibat dalam inisiatif sosial dan lingkungan, termasuk kegiatan amal dan program keberlanjutan.",
            "Keragaman dan inklusi adalah prioritas di perusahaan kami, dan kami berusaha menciptakan lingkungan yang ramah dan inklusif untuk semua."
        ]
    },
    "achievements": {
        "en": [
            "Some of our notable achievements include winning the 'Best Company of the Year' award and being recognized for our innovative products.",
            "We have contributed to the industry by introducing groundbreaking technologies and solutions.",
            "Our community involvement initiatives have been well-received, and we continue to make a positive impact in the community.",
            "We have several success stories and case studies that demonstrate the effectiveness of our products and services."
        ],
        "id": [
            "Beberapa pencapaian mencolok kami termasuk memenangkan penghargaan 'Perusahaan Terbaik Tahun Ini' dan diakui atas produk-produk inovatif kami.",
            "Kami telah berkontribusi pada industri dengan memperkenalkan teknologi dan solusi yang revolusioner.",
            "Inisiatif keterlibatan kami dalam komunitas telah diterima dengan baik, dan kami terus membuat dampak positif di komunitas.",
            "Kami memiliki beberapa cerita sukses dan studi kasus yang menunjukkan efektivitas produk dan layanan kami."
        ]
    },
    "future": {
        "en": [
            "Our future goals include expanding into new markets and introducing innovative products.",
            "We envision our company evolving into a global leader in our industry.",
            "We are excited about upcoming projects that will push the boundaries of technology and innovation.",
            "To stay competitive, we plan to invest in research and development and stay ahead of market trends."
        ],
        "id": [
            "Tujuan masa depan kami termasuk memperluas ke pasar baru dan memperkenalkan produk-produk inovatif.",
            "Kami memvisualisasikan perusahaan kami berkembang menjadi pemimpin global di industri kami.",
            "Kami bersemangat tentang proyek-proyek mendatang yang akan mendorong batas-batas teknologi dan inovasi.",
            "Untuk tetap kompetitif, kami berencana untuk berinvestasi dalam penelitian dan pengembangan dan tetap berada di depan tren pasar."
        ]
    },
    "greetings":{
         "en": [
        "Hello! How can I assist you today?",
        "Hi there! What can I do for you?",
        "Hey! How may I help you?",
        "Good day! How can I be of service?",
        "Greetings! What brings you here?"
    ],
    "id": [
        "Halo! Bagaimana saya bisa membantu Anda hari ini?",
        "Hai! Ada yang bisa saya bantu?",
        "Halo! Apa yang bisa saya bantu?",
        "Selamat hari! Bagaimana saya bisa membantu?",
        "Halo! Ada yang bisa saya bantu?"
    ]
    }
}

# Kamus pertanyaan
questions =  {
    "company": {
        "en": [
            "Can you tell me about your company?",
            "What is the history of your company?",
            "How long has your company been in business?",
            "What sets your company apart from others?",
            "What is your company's mission statement?",
            "Can you describe the culture at your company?"
        ],
        "id": [
            "Bisakah Anda ceritakan tentang perusahaan Anda?",
            "Apa sejarah perusahaan Anda?",
            "Berapa lama perusahaan Anda beroperasi?",
            "Apa yang membedakan perusahaan Anda dari yang lain?",
            "Apa pernyataan misi perusahaan Anda?",
            "Bisakah Anda mendeskripsikan budaya di perusahaan Anda?"
        ]
    },
    "team": {
        "en": [
            "Who are the key members of your team?",
            "What expertise does your team have?",
            "Can you introduce me to your team?",
            "How does your team collaborate on projects?",
            "What kind of training or development opportunities do you offer your team?"
        ],
        "id": [
            "Siapa anggota kunci dari tim Anda?",
            "Apa keahlian yang dimiliki oleh tim Anda?",
            "Bisakah Anda memperkenalkan saya kepada tim Anda?",
            "Bagaimana tim Anda berkolaborasi dalam proyek?",
            "Apa jenis pelatihan atau kesempatan pengembangan yang Anda tawarkan kepada tim Anda?"
        ]
    },
    "values": {
        "en": [
            "What are the core values of your company?",
            "How do you ensure your company valsues are reflected in your work?",
            "Are there any social or environmental initiatives your company is involved in?",
            "How do you prioritize diversity and inclusion in your company?",
            "Do you have a code of conduct or ethics policy for your company?"
        ],
        "id": [
            "Apa nilai inti dari perusahaan Anda?",
            "Bagaimana Anda memastikan nilai-nilai perusahaan Anda tercermin dalam pekerjaan Anda?",
            "Apakah ada inisiatif sosial atau lingkungan yang perusahaan Anda ikuti?",
            "Bagaimana Anda memprioritaskan keragaman dan inklusi di perusahaan Anda?",
            "Apakah Anda memiliki kode etik atau kebijakan etika untuk perusahaan Anda?"
        ]
    },
    "achievements": {
        "en": [
            "What are some notable achievements or milestones of your company?",
            "Can you share any awards or recognition your company has received?",
            "How has your company contributed to the industry or community?",
            "Do you have any success stories or case studies from your company?"
        ],
        "id": [
            "Apa pencapaian atau tonggak bersejarah yang mencolok dari perusahaan Anda?",
            "Bisakah Anda berbagi penghargaan atau pengakuan yang diterima perusahaan Anda?",
            "Bagaimana perusahaan Anda berkontribusi pada industri atau komunitas?",
            "Apakah Anda memiliki cerita sukses atau studi kasus dari perusahaan Anda?"
        ]
    },
    "future": {
        "en": [
            "What are your company's future goals or plans?",
            "How do you envision your company evolving in the future?",
            "Are there any upcoming projects or initiatives your company is excited about?",
            "How do you plan to stay competitive in the market?"
        ],
        "id": [
            "Apa tujuan atau rencana masa depan perusahaan Anda?",
            "Bagaimana Anda memvisualisasikan perkembangan perusahaan Anda di masa depan?",
            "Apakah ada proyek atau inisiatif mendatang yang membuat perusahaan Anda bersemangat?",
            "Bagaimana Anda berencana untuk tetap kompetitif di pasar?"
        ]
    },
    "greetings":{
      "en": [
        "Hi!",
        "Hello!",
        "Hey there!",
        "Good morning!",
        "Good afternoon!",
        "Good evening!",
        "What's up?",
        "How are you?",
        "How's your day?",
        "Nice to meet you."
    ],
    "id": [
        "Halo!",
        "Hai!",
        "Selamat pagi!",
        "Selamat siang!",
        "Selamat sore!",
        "Apa kabar?",
        "Bagaimana kabarmu?",
        "Bagaimana hari Anda?",
        "Senang bertemu dengan Anda."
    ]
    }
    
}

# def load_data(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#     return data

# # Load data from JSON file
# data = load_data('dataset.json')

# answers = data.get('answers', {})
# questions = data.get('questions', {})
# Fungsi untuk membersihkan teks
def clean_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english') + stopwords.words('indonesian'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return lemmatized_tokens

# Fungsi untuk mengekstrak fitur dari teks
def extract_features(text):
    features = {}
    for word in clean_text(text):
        features[word] = True
    return features

# Memproses data pelatihan
training_data = []
for intent, questions_dict in questions.items():
    for lang, question_list in questions_dict.items():
        for question in question_list:
            features = extract_features(question)
            training_data.append((features, intent))

# Melatih model Naive Bayes
classifier = NaiveBayesClassifier.train(training_data)

def detect_language(text):
    try:
        lang = detect(text)
    except:
        lang = 'en'  # Default to English if language detection fails
    return lang

def get_answer(question_text):
    lang = detect_language(question_text)
    features = extract_features(question_text)
    intent = classifier.classify(features)
    answers_list = answers[intent][lang]
    random.shuffle(answers_list)
    return answers_list[0] 
# Contoh penggunaan
exit_keywords = ["exit", "quit", "close"]

while True:
    question = input("Silakan masukkan pertanyaan Anda atau ketik 'exit' untuk keluar: ")
    
    # Keluar dari loop jika pengguna memasukkan kata kunci keluar
    if question.lower() in exit_keywords:
        print("Terima kasih telah menggunakan layanan kami.")
        break
    print(get_answer(question))

# def extract_features(text):
#     features = {}
#     for word in clean_text(text):
#         features[word] = True
#         # Menambahkan fitur untuk kata-kata yang mirip (fuzzy matching) dari semua topik
#         for topic, lang_questions in questions.items():
#             for lang, question_list in lang_questions.items():
#                 similar_words = process.extract(word, clean_text(' '.join(question_list)), limit=3)
#                 for similar_word, _ in similar_words:
#                     features[similar_word] = True
#     return features
