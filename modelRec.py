from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect
from nltk.classify import NaiveBayesClassifier
import random

# Data pertanyaan dan jawaban
questions ={
    "company": {
    "historyCompany": {
        "en": [
            "What is the history of your company?",
            "Can you tell me about your history company?",
            "Let me know about your history company?"
        ],
        "id": [
            "Apa sejarah perusahaan Anda?",
            "Bisakah anda katakan pada saya mengenai sejarah perusahaan Anda?",
            "Biarkan saya mengetahui mengenai sejarah perusahaan Anda?"
        ]
    },
    "visionAndMission": {
        "en": [
            "What is your company's vision and mission?",
            "What is your company's vision?",
            "What is your company's mission?",
            "Can you tell me about your company's vision and mission?",
            "What are the core values of your company?"
        ],
        "id": [
            "Apa visi dan misi perusahaan Anda?",
            "Bisakah Anda memberitahu saya tentang visi dan misi perusahaan Anda?",
            "Apa nilai inti dari perusahaan Anda?"
        ]
    },
    "cultureCompany": {
        "en": [
            "Can you describe the culture at your company?",
            "What is the work culture like at your company?",
            "How would you define the culture of your company?"
        ],
        "id": [
            "Bisakah Anda mendeskripsikan budaya di perusahaan Anda?",
            "Bagaimana budaya kerja di perusahaan Anda?",
            "Bagaimana Anda mendefinisikan budaya perusahaan Anda?"
        ]
    },
    "productsServices": {
        "en": [
            "What products or services does your company offer?",
            "Can you tell me about the products or services your company provides?",
            "What are the main products or services of your company?"
        ],
        "id": [
            "Produk atau layanan apa yang ditawarkan perusahaan Anda?",
            "Bisakah Anda memberitahu saya tentang produk atau layanan yang perusahaan Anda sediakan?",
            "Apa produk atau layanan utama dari perusahaan Anda?"
        ]
    },
    "teamCompany": {
        "en": [
            "Can you introduce me to your team?",
            "Who are the key members of your team?",
            "Tell me about the people who work at your company."
        ],
        "id": [
            "Bisakah Anda memperkenalkan saya kepada tim Anda?",
            "Siapa saja anggota kunci tim Anda?",
            "Ceritakan tentang orang-orang yang bekerja di perusahaan Anda."
        ]
    },
  
    "futureCompany": {
            "en": [
                "Our company is planning to expand its services globally and establish itself as a key player in the industry.",
                "We have several upcoming projects, including the development of new software solutions and partnerships with leading companies.",
                "In the future, we aim to be at the forefront of innovation, continuously improving our products and services to meet the evolving needs of our clients."
            ],
            "id": [
                "Perusahaan kami berencana untuk memperluas layanannya secara global dan menjadikan dirinya sebagai pemain kunci di industri ini.",
                "Kami memiliki beberapa proyek mendatang, termasuk pengembangan solusi perangkat lunak baru dan kemitraan dengan perusahaan-perusahaan terkemuka.",
                "Di masa depan, kami bertujuan untuk menjadi yang terdepan dalam inovasi, terus meningkatkan produk dan layanan kami untuk memenuhi kebutuhan yang terus berkembang dari klien kami."
            ]
    }
}
}

answers = {
    "historyCompany": {
        "en": [
            "Our company, Mangcoding, was founded in 2022 by Nugraha, who has more than 10 years of experience in website development. Since its inception, Mangcoding has been dedicated to providing high-quality website development services.",
            "Experiencing in Web Development more than 10 Years, Nugraha as a Project Manager in Mangcoding always get offer to work with company to build and manage their website. With his background as developer with US Company, he decide to create team in 2022."
        ],
        "id": [
            "Perusahaan kami, Mangcoding, didirikan pada tahun 2022 oleh Nugraha, yang memiliki pengalaman lebih dari 10 tahun di bidang pengembangan website. Sejak awal berdirinya, Mangcoding telah berkomitmen untuk menyediakan layanan pengembangan website berkualitas tinggi.",
            "Nugraha, yang memiliki pengalaman lebih dari 10 tahun di bidang pengembangan website, menjabat sebagai Project Manager di Mangcoding dan selalu mendapatkan tawaran untuk bekerja dengan perusahaan untuk membangun dan mengelola website mereka. Dengan latar belakangnya sebagai pengembang di perusahaan Amerika Serikat, ia memutuskan untuk membentuk tim pada tahun 2022."
        ]
    },
    "visionAndMission":{
        "en": [
        "Our vision at Mangcoding is to establish a global leadership position by offering services that are utilized worldwide. Our mission is to simplify performance monitoring, analysis, and enhancement for every individual within an organization, accessible from any device, anywhere.",
        "At Mangcoding, we aspire to lead globally by providing services used worldwide. Our mission is to streamline performance monitoring, analysis, and improvement for all employees, accessible from any device, anywhere."
        ],
        "id": [
            "Visi kami di Mangcoding adalah menjadi pemimpin global dengan menawarkan layanan yang digunakan di seluruh dunia. Misi kami adalah menyederhanakan pemantauan, analisis, dan peningkatan kinerja bagi setiap individu dalam sebuah organisasi, dapat diakses dari perangkat mana pun, di mana pun.",
            "Di Mangcoding, kami bermimpi memimpin secara global dengan menyediakan layanan yang digunakan di seluruh dunia. Misi kami adalah menyederhanakan pemantauan, analisis, dan peningkatan kinerja untuk semua karyawan, dapat diakses dari perangkat mana pun, di mana pun."
        ]
    },
    "cultureCompany":{
        "en": [
            "Our company culture is centered around a sense of family, fostering collaboration and innovation. Our work hours start at 8 AM WIB (1 AM GMT) and end at 5 PM WIB (10 AM GMT), promoting work-life balance and productivity.",
            "At Mangcoding, we pride ourselves on our familial culture, encouraging collaboration and innovation. Our office hours are from 8 AM to 5 PM WIB (1 AM to 10 AM GMT), ensuring a healthy work-life balance and optimal productivity.",
        ],
        "id": [
            "Budaya perusahaan kami berpusat pada rasa kekeluargaan, mendorong kolaborasi dan inovasi. Jam kerja kami dimulai pukul 8 pagi WIB dan berakhir pukul 5 sore WIB, mempromosikan keseimbangan antara kehidupan kerja dan produktivitas.",
            "Di Mangcoding, kami bangga dengan budaya kekeluargaan kami, mendorong kolaborasi dan inovasi. Jam kerja kami adalah dari pukul 8 pagi hingga 5 sore WIB, memastikan keseimbangan yang sehat antara kehidupan kerja dan produktivitas yang optimal."
        ]
    },
    "teamCompany":{
        "en": [
            "Our team at Mangcoding is a diverse group of talented individuals who are passionate about creating innovative solutions. Led by our experienced project manager, Nugraha, each member brings unique skills and perspectives to the table. From developers to designers, our team works collaboratively to deliver high-quality results for our clients.",
            "At Mangcoding, we pride ourselves on our familial culture, encouraging collaboration and innovation. Our office hours are from 8 AM to 5 PM WIB (1 AM to 10 AM GMT), ensuring a healthy work-life balance and optimal productivity.",
        ],
        "id": [
            "Budaya perusahaan kami berpusat pada rasa kekeluargaan, mendorong kolaborasi dan inovasi. Jam kerja kami dimulai pukul 8 pagi WIB dan berakhir pukul 5 sore WIB, mempromosikan keseimbangan antara kehidupan kerja dan produktivitas.",
            "Di Mangcoding, kami bangga dengan budaya kekeluargaan kami, mendorong kolaborasi dan inovasi. Jam kerja kami adalah dari pukul 8 pagi hingga 5 sore WIB, memastikan keseimbangan yang sehat antara kehidupan kerja dan produktivitas yang optimal."
        ]
    },
    "futureCompany":{
        "en": [
            "Our company is planning to expand its services globally and establish itself as a key player in the industry. We have several upcoming projects, including the development of new software solutions and partnerships with leading companies. In the future, we aim to be at the forefront of innovation, continuously improving our products and services to meet the evolving needs of our clients.",
            "We are focused on enhancing our technology and services to stay ahead of the competition. This includes investing in research and development to create cutting-edge solutions for our clients."
            "In the coming years, we aim to strengthen our partnerships with other industry leaders and explore new avenues for growth and development."
        ],
        "id": [
            "Perusahaan kami berencana untuk memperluas layanannya secara global dan menjadikan dirinya sebagai pemain kunci di industri ini. Kami memiliki beberapa proyek mendatang, termasuk pengembangan solusi perangkat lunak baru dan kemitraan dengan perusahaan-perusahaan terkemuka. Di masa depan, kami bertujuan untuk menjadi yang terdepan dalam inovasi, terus meningkatkan produk dan layanan kami untuk memenuhi kebutuhan yang terus berkembang dari klien kami.",
            "Kami fokus pada peningkatan teknologi dan layanan kami untuk tetap unggul dari kompetisi. Ini termasuk investasi dalam penelitian dan pengembangan untuk menciptakan solusi terkini bagi klien kami.",
            "Dalam beberapa tahun mendatang, kami bertujuan untuk memperkuat kemitraan kami dengan pemimpin industri lainnya dan mengeksplorasi jalur baru untuk pertumbuhan dan pengembangan."
            
        ]
    },
}


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
for main_topic, sub_topics in questions.items():
    for sub_topic, questions_dict in sub_topics.items():
        for lang, question_list in questions_dict.items():
            for question in question_list:
                features = extract_features(question)
                features['topic'] = main_topic  # Menambahkan main topic sebagai fitur
                training_data.append((features, sub_topic))

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
    main_topic = intent
    answers_list = answers[intent][lang]
    random.shuffle(answers_list)
    
    # Memilih satu pertanyaan dari setiap sub topik selain sub topik dari intent yang sedang ditanyakan
    recommended_questions = []
    for topic, sub_topics in questions.items():
        if topic == main_topic:
            continue
        for sub_topic, questions_dict in sub_topics.items():
            if sub_topic == intent:
                continue
            recommended_questions.append(random.choice(questions_dict[lang]))
    
    return main_topic, answers_list[0], recommended_questions

# Contoh penggunaan
exit_keywords = ["exit", "quit", "close"]

while True:
    question = input("Silakan masukkan pertanyaan Anda atau ketik 'exit' untuk keluar: ")

    # Keluar dari loop jika pengguna memasukkan kata kunci keluar
    if question.lower() in exit_keywords:
        print("Terima kasih telah menggunakan layanan kami.")
        break
    main_topic, answer, recommended_questions = get_answer(question)
    print("Main Topic:", main_topic)
    print("Answer:", answer)
    print("Recommended Questions:")
    for i, q in enumerate(recommended_questions, 1):
        print(f"{i}. {q}")

