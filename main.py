import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
import plotly.express as px
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="ML Spam Detector",
    page_icon="📱",
    layout="wide"
)

# Inyección de HTML/CSS/JS personalizado
st.markdown("""
    <style>
        /* Variables Globales */
        :root {
            --primary-color: #2E3192;
            --secondary-color: #1B998B;
            --text-color: #2C3E50;
            --background-color: #F8F9FA;
            --card-background: #FFFFFF;
            --success-color: #28a745;
            --warning-color: #ffc107;
            --danger-color: #dc3545;
        }

        /* Estilos Globales */
        body {
            font-family: 'Inter', sans-serif;
            color: var(--text-color);
            background-color: var(--background-color);
            margin: 0;
            padding: 0;
        }

        /* Navegación */
        .nav-wrapper {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: var(--card-background);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            z-index: 1000;
            padding: 1rem 2rem;
        }

        .nav-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1200px;
            margin: 0 auto;
        }

        .nav-links a {
            color: var(--text-color);
            text-decoration: none;
            margin-left: 2rem;
            transition: color 0.3s ease;
        }

        .nav-links a:hover {
            color: var(--primary-color);
        }

        /* Header */
        .header-wrapper {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 4rem 2rem 2rem;
            text-align: center;
            margin-bottom: 2rem;
        }

        .author-info {
            margin-top: 1.5rem;
        }

        .social-links {
            margin-top: 1rem;
        }

        .social-links a {
            color: white;
            text-decoration: none;
            margin: 0 1rem;
            padding: 0.5rem 1rem;
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 20px;
            transition: all 0.3s ease;
        }

        .social-links a:hover {
            background: rgba(255,255,255,0.1);
            transform: translateY(-2px);
        }

        /* Contenido Principal */
        .main-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 2rem;
        }

        .card {
            background: var(--card-background);
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 2rem;
            margin-bottom: 2rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }

        /* Botones */
        .stButton>button {
            background: var(--primary-color);
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 1rem;
        }

        .stButton>button:hover {
            background: var(--secondary-color);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }

        /* Métricas y Resultados */
        .metric-card {
            background: var(--card-background);
            padding: 1.5rem;
            border-radius: 8px;
            text-align: center;
            margin: 1rem 0;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary-color);
        }

        .metric-label {
            color: var(--text-color);
            margin-top: 0.5rem;
        }

        /* Formularios */
        .stTextArea>div>div>textarea {
            border-radius: 8px;
            border: 2px solid #e0e0e0;
            padding: 1rem;
            font-size: 1rem;
            transition: border-color 0.3s ease;
        }

        .stTextArea>div>div>textarea:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 2px rgba(46,49,146,0.1);
        }

        /* Footer */
        .footer-wrapper {
            background: var(--text-color);
            color: white;
            padding: 3rem 2rem;
            margin-top: 4rem;
        }

        .footer-content {
            max-width: 1200px;
            margin: 0 auto;
            text-align: center;
        }

        .footer-links {
            margin-top: 1.5rem;
        }

        .footer-links a {
            color: white;
            text-decoration: none;
            margin: 0 1rem;
            opacity: 0.8;
            transition: opacity 0.3s ease;
        }

        .footer-links a:hover {
            opacity: 1;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .nav-content {
                flex-direction: column;
                text-align: center;
            }

            .nav-links {
                margin-top: 1rem;
            }

            .nav-links a {
                display: block;
                margin: 0.5rem 0;
            }

            .header-wrapper {
                padding: 3rem 1rem 1.5rem;
            }

            .main-content {
                padding: 0 1rem;
            }

            .social-links a {
                display: inline-block;
                margin: 0.5rem;
            }
        }
    </style>

    <!-- Navegación -->
    <div class="nav-wrapper">
        <div class="nav-content">
            <div class="logo">📱 ML Spam Detector</div>
            <div class="nav-links">
                <a href="#dataset">Dataset</a>
                <a href="#model">Modelo</a>
                <a href="#test">Prueba</a>
            </div>
        </div>
    </div>

    <!-- Header -->
    <div class="header-wrapper">
        <h1>Detector de Spam con Machine Learning</h1>
        <div class="author-info">
            <h2>Desarrollado por Marco Mayta</h2>
            <div class="social-links">
                <a href="https://github.com/Maqui2404" target="_blank">
                    <i class="fab fa-github"></i> GitHub
                </a>
                <a href="https://linkedin.com/in/marco-mayta-835781170/" target="_blank">
                    <i class="fab fa-linkedin"></i> LinkedIn
                </a>
                <a href="https://maqui2404.github.io/PortafolioMarco.github.io/" target="_blank">
                    <i class="fas fa-user"></i> Portafolio
                </a>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Funciones de carga y procesamiento de datos


@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])
    return df


# Sección del Dataset
st.markdown('<div id="dataset" class="main-content">', unsafe_allow_html=True)
st.header("Análisis del Dataset")
with st.expander("Ver detalles del dataset", expanded=True):
    df = load_data()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Muestra de Mensajes")
        st.dataframe(df.head(), use_container_width=True)

    with col2:
        st.markdown("### Distribución de Clases")
        fig = px.pie(values=df['label'].value_counts().values,
                     names=df['label'].value_counts().index,
                     title='Proporción de Mensajes Spam vs. Ham')
        st.plotly_chart(fig, use_container_width=True)

# Preparación y procesamiento
st.markdown('<div id="model" class="main-content">', unsafe_allow_html=True)
st.header("Preparación y Procesamiento")
st.markdown("""
#### Proceso de Preparación de Datos:
1. **División del Dataset**: 
   - 80% para entrenamiento
   - 20% para pruebas
   
2. **Vectorización TF-IDF**:
   - Convierte texto en vectores numéricos
   - Considera la frecuencia e importancia de las palabras
   - Elimina palabras comunes sin valor predictivo
""")

X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Entrenamiento del modelo
st.header("Entrenamiento del Modelo")
st.markdown("""
#### Algoritmo Naive Bayes
El clasificador Naive Bayes es especialmente efectivo para la clasificación de texto porque:
- Es rápido y eficiente
- Funciona bien con datos de alta dimensionalidad
- Requiere menos datos de entrenamiento
- Es robusto ante características irrelevantes
""")

train_model = st.button("Entrenar Modelo", key="train")

if train_model:
    with st.spinner("Entrenando el modelo..."):
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)
        time.sleep(1)

        # Evaluación
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        st.success("✅ ¡Modelo entrenado exitosamente!")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Precisión del Modelo", f"{accuracy:.2%}")
        with col2:
            st.metric("Verdaderos Positivos", conf_matrix[1, 1])
        with col3:
            st.metric("Verdaderos Negativos", conf_matrix[0, 0])

        st.markdown("### Reporte Detallado")
        st.code(classification_report(y_test, y_pred))

        st.session_state["model"] = model
        st.session_state["vectorizer"] = vectorizer

# Sección de prueba
st.markdown('<div id="test" class="main-content">', unsafe_allow_html=True)
st.header("Prueba el Modelo")
st.markdown("""
### Predicción en Tiempo Real
Ingresa un mensaje de texto para clasificarlo como spam o no spam (ham).
El modelo analizará las características del texto y realizará una predicción basada en los patrones aprendidos.
""")

with st.form("prediction_form"):
    user_input = st.text_area("📝 Ingresa un mensaje de texto:", height=100)
    submit_button = st.form_submit_button("🔍 Analizar Mensaje")

if submit_button:
    if "model" not in st.session_state:
        st.warning(
            "⚠️ Primero debes entrenar el modelo haciendo clic en el botón 'Entrenar Modelo'.")
    else:
        model = st.session_state["model"]
        vectorizer = st.session_state["vectorizer"]

        user_input_vec = vectorizer.transform([user_input])
        prediction = model.predict(user_input_vec)
        prob = model.predict_proba(user_input_vec)[0]

        st.markdown("### Resultado del Análisis")
        col1, col2 = st.columns(2)

        with col1:
            if prediction[0] == "spam":
                st.error("🚫 Mensaje clasificado como SPAM")
            else:
                st.success("✅ Mensaje clasificado como HAM (No Spam)")

        with col2:
            st.markdown(f"""
            **Probabilidades:**
            - Spam: {prob[1]:.2%}
            - Ham: {prob[0]:.2%}
            """)

# Footer
st.markdown("""
    <div class="footer-wrapper">
        <div class="footer-content">
            <p>© 2025 ML Spam Detector. Todos los derechos reservados.</p>
            <p>Desarrollado por Marco Mayta usando Streamlit, Scikit-learn y Python</p>
            <div class="footer-links">
                <a href="#privacy">Política de Privacidad</a>
                <a href="#terms">Términos de Servicio</a>
                <a href="#contact">Contacto</a>
            </div>
        </div>
    </div>

    <script>
        // Smooth scrolling
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({
                    behavior: 'smooth'
                });
            });
        });

        // Animaciones de scroll para las cards
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = 1;
                    entry.target.style.transform = 'translateY(0)';
                }
            });
        });

        document.querySelectorAll('.card').forEach((card) => {
            card.style.opacity = 0;
            card.style.transform = 'translateY(20px)';
            card.style.transition = 'all 0.6s ease-out';
            observer.observe(card);
        });

        // Animación para métricas
        const animateValue = (element, start, end, duration) => {
            let startTimestamp = null;
            const step = (timestamp) => {
                if (!startTimestamp) startTimestamp = timestamp;
                const progress = Math.min((timestamp - startTimestamp) / duration, 1);
                const value = Math.floor(progress * (end - start) + start);
                element.textContent = value.toFixed(1) + '%';
                if (progress < 1) {
                    window.requestAnimationFrame(step);
                }
            };
            window.requestAnimationFrame(step);
        };

        // Inicializar las animaciones cuando los elementos sean visibles
        const metricObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const valueElement = entry.target.querySelector('.metric-value');
                    if (valueElement) {
                        const finalValue = parseFloat(valueElement.getAttribute('data-value'));
                        animateValue(valueElement, 0, finalValue, 1000);
                    }
                }
            });
        });

        document.querySelectorAll('.metric-card').forEach((metric) => {
            metricObserver.observe(metric);
        });

        // Mejorar la experiencia de navegación
        window.addEventListener('scroll', () => {
            const nav = document.querySelector('.nav-wrapper');
            if (window.scrollY > 100) {
                nav.style.backgroundColor = 'rgba(255, 255, 255, 0.95)';
                nav.style.backdropFilter = 'blur(5px)';
            } else {
                nav.style.backgroundColor = 'var(--card-background)';
                nav.style.backdropFilter = 'none';
            }
        });

        // Inicializar tooltips
        const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
        const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
    </script>

    <!-- Font Awesome para iconos -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" />
""", unsafe_allow_html=True)
