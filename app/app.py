import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os

# ── Configuration ──────────────────────────────────────────────────────────────
CLASS_NAMES = ['Kus_Gozu_Mantari', 'Pas_Akari', 'Saglikli']
CLASS_FR    = ['Œil de Paon',       'Acariose',   'Feuille Saine']
IMG_SIZE    = 224

CLASS_INFO = {
    'Kus_Gozu_Mantari': {
        'nom'         : 'Œil de Paon',
        'nom_sci'     : 'Cycloconium oleaginum',
        'emoji'       : '🟡',
        'couleur'     : '#d4900a',
        'bg'          : '#fff8e6',
        'border'      : '#f5c842',
        'gravite'     : 'Modérée',
        'gravite_icon': '⚠️',
        'gravite_color': '#d4900a',
        'description' : "Maladie fongique très répandue dans les oliveraies méditerranéennes. Elle se développe en conditions humides et fraîches, principalement en automne et au printemps.",
        'symptomes'   : [
            ('👁️', 'Taches circulaires jaune-verdâtre sur le dessus des feuilles'),
            ('🟤', 'Halo brun-grisâtre velouté sous les feuilles'),
            ('🍂', 'Chute prématurée et massive des feuilles'),
            ('📉', 'Réduction significative de la production d\'olives'),
            ('🌿', 'Rameaux affaiblis et croissance ralentie'),
        ],
        'medicaments' : [
            {
                'nom'   : 'Bouillie Bordelaise',
                'type'  : 'Fongicide cuivrique',
                'dose'  : '0.5 - 1%',
                'moment': 'Automne (chute des feuilles) + Printemps',
                'icon'  : '🔵',
            },
            {
                'nom'   : 'Hydroxyde de cuivre',
                'type'  : 'Fongicide de contact',
                'dose'  : '1.5 - 2 kg/ha',
                'moment': 'Avant les pluies d\'automne',
                'icon'  : '🟢',
            },
            {
                'nom'   : 'Oxychlorure de cuivre',
                'type'  : 'Fongicide préventif',
                'dose'  : '250-300 g/hl',
                'moment': 'Début automne et fin hiver',
                'icon'  : '🟤',
            },
        ],
        'prevention'  : [
            '✂️ Tailler régulièrement pour aérer le feuillage',
            '🚫 Éviter l\'excès d\'irrigation',
            '🗑️ Ramasser et brûler les feuilles tombées',
            '📅 Traitement préventif chaque automne',
        ],
        'urgence'     : 'Traiter dans les 2 semaines suivant la détection',
    },
    'Pas_Akari': {
        'nom'         : 'Acariose',
        'nom_sci'     : 'Aculus olearius',
        'emoji'       : '🔴',
        'couleur'     : '#c0392b',
        'bg'          : '#fff0ee',
        'border'      : '#e74c3c',
        'gravite'     : 'Élevée',
        'gravite_icon': '🚨',
        'gravite_color': '#c0392b',
        'description' : "Infestation par l'acarien microscopique Aculus olearius. Ces acariens invisibles à l'œil nu colonisent les feuilles et perturbent gravement la photosynthèse, pouvant réduire la récolte de 20 à 40%.",
        'symptomes'   : [
            ('🥈', 'Feuilles argentées ou bronzées (aspect métallique)'),
            ('🌀', 'Déformation et enroulement des feuilles'),
            ('📏', 'Réduction de la taille des jeunes pousses'),
            ('🔍', 'Présence visible de poussière brunâtre (excréments)'),
            ('🍂', 'Chute des feuilles dans les cas sévères'),
        ],
        'medicaments' : [
            {
                'nom'   : 'Soufre mouillable',
                'type'  : 'Acaricide naturel',
                'dose'  : '0.3 - 0.4%',
                'moment': 'Printemps (débourrement) + été',
                'icon'  : '🟡',
            },
            {
                'nom'   : 'Huile minérale blanche',
                'type'  : 'Acaricide suffocant',
                'dose'  : '1.5 - 2%',
                'moment': 'Hiver (stade dormant)',
                'icon'  : '⚪',
            },
            {
                'nom'   : 'Abamectine',
                'type'  : 'Acaricide systémique',
                'dose'  : '7.5 ml/hl',
                'moment': 'Printemps lors de la première détection',
                'icon'  : '🔴',
            },
            {
                'nom'   : 'Bifenazate',
                'type'  : 'Acaricide sélectif',
                'dose'  : '40 ml/hl',
                'moment': 'Été — pic d\'infestation',
                'icon'  : '🟠',
            },
        ],
        'prevention'  : [
            '🔍 Surveiller hebdomadairement les nouvelles pousses',
            '🌊 Maintenir une humidité optimale (éviter la sécheresse)',
            '🐞 Favoriser les prédateurs naturels (acariens utiles)',
            '💨 Assurer une bonne circulation d\'air dans l\'oliveraie',
        ],
        'urgence'     : 'URGENT — Traiter immédiatement, l\'infestation se propage rapidement',
    },
    'Saglikli': {
        'nom'         : 'Feuille Saine',
        'nom_sci'     : 'Olea europaea (sain)',
        'emoji'       : '🟢',
        'couleur'     : '#27ae60',
        'bg'          : '#edfbf3',
        'border'      : '#2ecc71',
        'gravite'     : 'Aucune',
        'gravite_icon': '✅',
        'gravite_color': '#27ae60',
        'description' : "Votre olivier est en excellente santé ! Les feuilles présentent toutes les caractéristiques d'une plante vigoureuse et bien entretenue.",
        'symptomes'   : [
            ('💚', 'Couleur vert foncé uniforme et brillante'),
            ('✨', 'Surface lisse et cireuse sans taches'),
            ('📐', 'Forme lancéolée normale et symétrique'),
            ('💪', 'Texture ferme et résistante'),
            ('🌱', 'Croissance active et régulière'),
        ],
        'medicaments' : [
            {
                'nom'   : 'Engrais foliaire NPK',
                'type'  : 'Nutrition préventive',
                'dose'  : '2-3 kg/hl',
                'moment': 'Printemps et automne',
                'icon'  : '🌿',
            },
            {
                'nom'   : 'Cuivre préventif (faible dose)',
                'type'  : 'Protection fongique',
                'dose'  : '0.3%',
                'moment': 'Avant la saison des pluies',
                'icon'  : '🛡️',
            },
        ],
        'prevention'  : [
            '💧 Irrigation régulière et adaptée à la saison',
            '✂️ Taille annuelle pour maintenir l\'aération',
            '🌱 Fertilisation équilibrée (N-P-K)',
            '👁️ Surveillance mensuelle préventive',
            '🧹 Entretien du sol autour des arbres',
        ],
        'urgence'     : 'Aucun traitement nécessaire — Continuer la surveillance préventive',
    },
}

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='OliveGuard — Détection Maladies Olivier',
    page_icon='🫒',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --olive-dark:   #1a2e0d;
    --olive-deep:   #2d4a1e;
    --olive-mid:    #4a7c2f;
    --olive-light:  #8ab44a;
    --olive-pale:   #d4e8b0;
    --cream:        #faf8f3;
    --gold:         #c9a84c;
    --gold-light:   #f0d080;
    --text:         #1a2e0d;
    --text-soft:    #5a7a4a;
}

* { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--cream);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a2e0d 0%, #2d4a1e 50%, #1a2e0d 100%) !important;
    border-right: 1px solid rgba(201,168,76,0.3);
}
section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.9) !important; }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    color: #c9a84c !important;
    font-family: 'Cormorant Garamond', serif !important;
}
section[data-testid="stSidebar"] .stMarkdown hr {
    border-color: rgba(201,168,76,0.3) !important;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #1a2e0d 0%, #2d4a1e 40%, #4a7c2f 70%, #8ab44a 100%);
    border-radius: 20px;
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(26,46,13,0.4);
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Ccircle cx='30' cy='30' r='20'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    pointer-events: none;
}
.hero-emoji { font-size: 3.5rem; margin-bottom: 0.5rem; display: block; }
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.5rem; font-weight: 700;
    color: white; margin: 0; line-height: 1;
    text-shadow: 0 4px 20px rgba(0,0,0,0.3);
    letter-spacing: -1px;
}
.hero-sub {
    color: rgba(255,255,255,0.8);
    font-size: 1rem; margin: 0.8rem 0 1.2rem;
    font-weight: 300; letter-spacing: 2px;
    text-transform: uppercase;
}
.hero-tags { display: flex; gap: 0.8rem; justify-content: center; flex-wrap: wrap; }
.hero-tag {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.25);
    color: white; padding: 0.4rem 1.2rem;
    border-radius: 30px; font-size: 0.82rem;
    backdrop-filter: blur(10px); font-weight: 500;
}

/* ── Upload Zone ── */
.upload-area {
    border: 2px dashed rgba(74,124,47,0.4);
    border-radius: 16px; padding: 2.5rem;
    text-align: center;
    background: linear-gradient(135deg, rgba(138,180,74,0.05), rgba(74,124,47,0.08));
    transition: all 0.3s ease; cursor: pointer;
}
.upload-icon { font-size: 3rem; margin-bottom: 0.8rem; }
.upload-text { color: #4a7c2f; font-size: 1rem; font-weight: 500; }
.upload-sub  { color: #8ab44a; font-size: 0.82rem; margin-top: 0.3rem; }

/* ── Diagnostic Card ── */
.diag-card {
    border-radius: 20px; padding: 2rem;
    margin-bottom: 1.5rem;
    border: 1.5px solid;
    position: relative; overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.08);
}
.diag-card::before {
    content: attr(data-emoji);
    position: absolute; right: 1.5rem; top: 1rem;
    font-size: 4rem; opacity: 0.12;
}
.diag-badge {
    display: inline-block; padding: 0.3rem 1rem;
    border-radius: 20px; font-size: 0.78rem;
    font-weight: 600; text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 0.8rem;
}
.diag-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.5rem; font-weight: 700;
    margin: 0; line-height: 1.1;
}
.diag-sci {
    font-style: italic; font-size: 0.85rem;
    opacity: 0.6; margin-top: 0.2rem;
}
.conf-row {
    display: flex; justify-content: space-between;
    align-items: center; margin: 1.2rem 0 0.4rem;
}
.conf-label { font-size: 0.82rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.6; }
.conf-value { font-size: 1.4rem; font-weight: 700; font-family: 'Cormorant Garamond', serif; }
.conf-bar-bg {
    background: rgba(0,0,0,0.08); border-radius: 10px;
    height: 8px; overflow: hidden;
}
.conf-bar-fill { height: 100%; border-radius: 10px; transition: width 1s ease; }

/* ── Prob Bars ── */
.prob-section { margin-top: 1.5rem; }
.prob-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.3rem; font-weight: 600;
    color: var(--olive-dark); margin-bottom: 1rem;
}
.prob-row {
    display: flex; align-items: center;
    gap: 1rem; margin: 0.6rem 0;
}
.prob-label {
    width: 120px; text-align: right;
    font-size: 0.85rem; color: #444;
    font-weight: 500;
}
.prob-bar-bg {
    flex: 1; background: rgba(0,0,0,0.06);
    border-radius: 8px; height: 24px; overflow: hidden;
    position: relative;
}
.prob-bar-fill {
    height: 100%; border-radius: 8px;
    display: flex; align-items: center;
    padding-left: 10px; min-width: 3px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}
.prob-pct { font-size: 0.8rem; font-weight: 700; color: white; }

/* ── Disease Detail Panel ── */
.panel {
    border-radius: 20px; padding: 2rem;
    margin-top: 1.5rem;
    border: 1.5px solid;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
}
.panel-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.6rem; font-weight: 700;
    margin-bottom: 1rem;
    display: flex; align-items: center; gap: 0.5rem;
}

/* ── Symptom Items ── */
.symptom-item {
    display: flex; align-items: flex-start; gap: 0.8rem;
    padding: 0.7rem 1rem; margin: 0.4rem 0;
    background: rgba(255,255,255,0.7);
    border-radius: 10px; font-size: 0.88rem;
    border-left: 3px solid transparent;
}
.symptom-icon { font-size: 1.1rem; flex-shrink: 0; }

/* ── Medicine Cards ── */
.med-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 1rem; margin-top: 1rem;
}
.med-card {
    background: white; border-radius: 14px;
    padding: 1.2rem; border: 1px solid rgba(0,0,0,0.08);
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    position: relative; overflow: hidden;
}
.med-card::before {
    content: attr(data-icon);
    position: absolute; right: 1rem; top: 0.8rem;
    font-size: 1.8rem; opacity: 0.15;
}
.med-type {
    font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 1px; font-weight: 600;
    opacity: 0.5; margin-bottom: 0.3rem;
}
.med-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.1rem; font-weight: 700;
    margin-bottom: 0.5rem; line-height: 1.2;
}
.med-info {
    display: flex; gap: 0.5rem; flex-wrap: wrap;
    margin-top: 0.5rem;
}
.med-tag {
    font-size: 0.72rem; padding: 0.2rem 0.6rem;
    border-radius: 6px; font-weight: 500;
}

/* ── Prevention ── */
.prev-item {
    padding: 0.6rem 1rem; margin: 0.3rem 0;
    background: rgba(255,255,255,0.6);
    border-radius: 8px; font-size: 0.87rem;
    border-left: 3px solid rgba(74,124,47,0.3);
}

/* ── Urgency Banner ── */
.urgency {
    padding: 1rem 1.5rem; border-radius: 12px;
    margin-top: 1.2rem; font-weight: 600;
    font-size: 0.9rem; display: flex;
    align-items: center; gap: 0.8rem;
}

/* ── Metric Boxes ── */
.metric-box {
    background: white; border-radius: 12px;
    padding: 1rem; text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    border: 1px solid rgba(74,124,47,0.1);
}
.metric-val {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.8rem; font-weight: 700;
    color: var(--olive-deep);
}
.metric-lbl {
    font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 1px; color: #888; margin-top: 0.2rem;
}

/* ── Guide Cards (bottom) ── */
.guide-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 1.5rem; margin-top: 1rem;
}
.guide-card {
    border-radius: 16px; padding: 1.8rem;
    border: 1.5px solid; text-align: center;
}
.guide-icon { font-size: 2.5rem; margin-bottom: 0.8rem; }
.guide-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.3rem; font-weight: 700; margin-bottom: 0.5rem;
}
.guide-text { font-size: 0.85rem; opacity: 0.75; line-height: 1.5; }

/* ── Section Headers ── */
.section-header {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem; font-weight: 700;
    color: var(--olive-dark);
    border-bottom: 2px solid var(--gold);
    padding-bottom: 0.5rem; margin: 2rem 0 1.5rem;
    display: inline-block;
}

/* ── Footer ── */
.footer {
    text-align: center; color: #999;
    font-size: 0.8rem; padding: 2rem 0 1rem;
    border-top: 1px solid rgba(74,124,47,0.15);
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫒 OliveGuard")
    st.markdown("---")
    st.markdown("### 📊 Modèle IA")
    st.markdown("""
**Architecture** : EfficientNetB4
**Accuracy** : 90.48%
**F1-Score** : 0.9184
**Dataset** : Zeytin 224×224
**Images entraînement** : 6 961
""")
    st.markdown("---")
    st.markdown("### 🌿 Classes détectées")
    for cls, info in CLASS_INFO.items():
        st.markdown(f"{info['emoji']} **{info['nom']}**")
        st.markdown(f"<small style='opacity:0.6;font-style:italic'>{info['nom_sci']}</small>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📋 Guide rapide")
    st.markdown("""
1. 📸 Importez une photo de feuille
2. 🔍 L'IA analyse en quelques secondes
3. 📊 Consultez le diagnostic complet
4. 💊 Suivez les recommandations
""")
    st.markdown("---")
    st.markdown("### ℹ️ À propos")
    st.markdown("""
Projet Deep Learning de détection automatique des maladies de l'olivier.

**Encadrant** : M. Abdallah Khemais
**Technologie** : TensorFlow / EfficientNetB4
""")

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-emoji">🫒</span>
    <h1 class="hero-title">OliveGuard</h1>
    <p class="hero-sub">Système de détection des maladies de l'olivier par Intelligence Artificielle</p>
    <div class="hero-tags">
        <span class="hero-tag">⚡ EfficientNetB4</span>
        <span class="hero-tag">🎯 Accuracy 90.48%</span>
        <span class="hero-tag">🌿 3 Classes</span>
        <span class="hero-tag">🔬 Deep Learning</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    candidates = [
        'olive_efficientnetb4_best.keras',
        'vgg16_final.keras',
        'vgg16_best_real.h5',
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                m = tf.keras.models.load_model(path, compile=False)
                m.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                return m, path
            except Exception as e:
                continue
    return None, None

model, model_path = load_model()

if model is None:
    st.error("❌ Modèle introuvable — placez `olive_efficientnetb4_best.keras` dans le dossier app/")
    st.stop()

# ── Grad-CAM ───────────────────────────────────────────────────────────────────
def make_gradcam(img_array, model):
    try:
        conv_layers = []
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                conv_layers.append(layer.name)
            elif hasattr(layer, 'layers'):
                for sub in layer.layers:
                    if isinstance(sub, tf.keras.layers.Conv2D):
                        conv_layers.append(sub.name)
        if not conv_layers:
            return None
        last_conv = conv_layers[-1]
        target = None
        for layer in model.layers:
            if layer.name == last_conv:
                target = layer; break
            elif hasattr(layer, 'layers'):
                for sub in layer.layers:
                    if sub.name == last_conv:
                        target = sub; break
        if target is None:
            return None
        grad_model = tf.keras.Model(inputs=model.inputs, outputs=[target.output, model.output])
        with tf.GradientTape() as tape:
            inputs = tf.cast(img_array, tf.float32)
            conv_out, preds = grad_model(inputs)
            tape.watch(conv_out)
            cls = tf.argmax(preds[0])
            ch  = preds[:, cls]
        grads = tape.gradient(ch, conv_out)
        pg    = tf.reduce_mean(grads, axis=(0,1,2))
        hm    = conv_out[0] @ pg[..., tf.newaxis]
        hm    = tf.squeeze(hm)
        hm    = tf.maximum(hm, 0) / (tf.math.reduce_max(hm) + 1e-8)
        return hm.numpy()
    except:
        return None

def overlay_gradcam(img_orig, heatmap, alpha=0.4):
    heat_r  = cv2.resize(heatmap, (img_orig.shape[1], img_orig.shape[0]))
    heat_c  = cv2.applyColorMap(np.uint8(255*heat_r), cv2.COLORMAP_JET)
    heat_c  = cv2.cvtColor(heat_c, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_orig, 1-alpha, heat_c, alpha, 0)
    return overlay, heat_r

# ── Main Layout ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.5], gap="large")

with col_left:
    st.markdown("### 📸 Importer une image")
    uploaded = st.file_uploader(
        "Photo feuille olivier",
        type=['jpg','jpeg','png'],
        label_visibility='collapsed'
    )
    if uploaded:
        img_pil     = Image.open(uploaded).convert('RGB')
        img_np      = np.array(img_pil)
        img_resized = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
        st.image(img_pil, caption="Image analysée", use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{img_pil.width}</div><div class="metric-lbl">Largeur px</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{img_pil.height}</div><div class="metric-lbl">Hauteur px</div></div>', unsafe_allow_html=True)
        with c3:
            size_kb = len(uploaded.getvalue()) // 1024
            st.markdown(f'<div class="metric-box"><div class="metric-val">{size_kb}</div><div class="metric-lbl">Taille KB</div></div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="upload-area">
            <div class="upload-icon">🌿</div>
            <div class="upload-text">Glissez une photo de feuille d'olivier</div>
            <div class="upload-sub">JPG, JPEG, PNG · Toute résolution</div>
        </div>
        """, unsafe_allow_html=True)

with col_right:
    if uploaded:
        with st.spinner('🔍 Analyse par l\'IA en cours...'):
            img_input  = np.expand_dims(img_resized.astype(np.float32), axis=0)
            probas     = model.predict(img_input, verbose=0)[0]
            pred_idx   = int(np.argmax(probas))
            pred_cls   = CLASS_NAMES[pred_idx]
            confidence = float(probas[pred_idx])
            info       = CLASS_INFO[pred_cls]

        # ── Diagnostic Card ────────────────────────────────────────────────
        st.markdown(f"""
        <div class="diag-card" data-emoji="{info['emoji']}"
             style="background:{info['bg']};border-color:{info['border']};">
            <div class="diag-badge"
                 style="background:{info['couleur']}22;color:{info['couleur']};">
                {info['gravite_icon']} Gravité : {info['gravite']}
            </div>
            <div class="diag-name" style="color:{info['couleur']}">
                {info['emoji']} {info['nom']}
            </div>
            <div class="diag-sci">{info['nom_sci']}</div>
            <div class="conf-row">
                <span class="conf-label">Confiance du modèle</span>
                <span class="conf-value" style="color:{info['couleur']}">{confidence*100:.1f}%</span>
            </div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{confidence*100:.1f}%;background:{info['couleur']};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probabilités ──────────────────────────────────────────────────
        st.markdown('<div class="prob-title">📊 Probabilités par classe</div>', unsafe_allow_html=True)
        colors = ['#d4900a', '#c0392b', '#27ae60']
        for i, (cls, fr, prob, color) in enumerate(zip(CLASS_NAMES, CLASS_FR, probas, colors)):
            pct    = float(prob) * 100
            bold   = 'font-weight:700;color:#1a2e0d;' if i == pred_idx else 'color:#666;'
            active = f'box-shadow:0 0 0 2px {color}30;' if i == pred_idx else ''
            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-label" style="{bold}">{CLASS_INFO[CLASS_NAMES[i]]['emoji']} {fr}</div>
                <div class="prob-bar-bg" style="{active}">
                    <div class="prob-bar-fill" style="width:{max(pct,2):.1f}%;background:linear-gradient(90deg,{color},{color}cc);">
                        <span class="prob-pct">{pct:.1f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Grad-CAM ──────────────────────────────────────────────────────
        st.markdown("### 🔥 Zones d'attention — Grad-CAM")
        with st.spinner('Génération de la carte thermique...'):
            heatmap = make_gradcam(img_input, model)
        if heatmap is not None:
            overlay, heat_r = overlay_gradcam(img_resized, heatmap)
            gc1, gc2 = st.columns(2)
            with gc1:
                st.image(heat_r, caption='Heatmap', use_container_width=True, clamp=True)
            with gc2:
                st.image(overlay, caption='Superposition', use_container_width=True)
        else:
            st.info("Grad-CAM non disponible pour ce modèle.")

    else:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                    height:500px;background:white;border-radius:20px;
                    border:1.5px dashed rgba(74,124,47,0.2);
                    box-shadow:0 4px 20px rgba(26,46,13,0.06);">
            <div style="font-size:5rem;margin-bottom:1rem">🔬</div>
            <div style="font-family:'Cormorant Garamond',serif;font-size:1.5rem;
                        color:#2d4a1e;font-weight:600;text-align:center;">
                Importez une image<br>pour démarrer l'analyse
            </div>
            <div style="color:#8ab44a;margin-top:0.5rem;font-size:0.9rem;">
                Diagnostic en temps réel par EfficientNetB4
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Disease Detail Section ─────────────────────────────────────────────────────
if uploaded and 'pred_cls' in dir():
    info = CLASS_INFO[pred_cls]
    st.markdown("---")
    st.markdown(f'<div class="section-header">📋 Rapport Complet — {info["nom"]}</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        # Description
        st.markdown(f"""
        <div class="panel" style="background:{info['bg']};border-color:{info['border']}40;">
            <div class="panel-title" style="color:{info['couleur']}">
                🔍 Description de la maladie
            </div>
            <p style="font-size:0.92rem;line-height:1.7;color:#333;margin:0">{info['description']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Symptômes
        st.markdown(f"""
        <div class="panel" style="background:{info['bg']};border-color:{info['border']}40;margin-top:1rem;">
            <div class="panel-title" style="color:{info['couleur']}">🩺 Symptômes identifiés</div>
        """, unsafe_allow_html=True)
        for icon, symptome in info['symptomes']:
            st.markdown(f"""
            <div class="symptom-item" style="border-left-color:{info['couleur']};">
                <span class="symptom-icon">{icon}</span>
                <span>{symptome}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        # Traitements / Médicaments
        st.markdown(f"""
        <div class="panel" style="background:{info['bg']};border-color:{info['border']}40;">
            <div class="panel-title" style="color:{info['couleur']}">💊 Traitements recommandés</div>
            <div class="med-grid">
        """, unsafe_allow_html=True)
        for med in info['medicaments']:
            st.markdown(f"""
            <div class="med-card" data-icon="{med['icon']}">
                <div class="med-type">{med['type']}</div>
                <div class="med-name">{med['nom']}</div>
                <div class="med-info">
                    <span class="med-tag" style="background:{info['couleur']}18;color:{info['couleur']}">
                        📏 {med['dose']}
                    </span>
                    <span class="med-tag" style="background:rgba(0,0,0,0.06);color:#555">
                        📅 {med['moment']}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

        # Prévention
        st.markdown(f"""
        <div class="panel" style="background:{info['bg']};border-color:{info['border']}40;margin-top:1rem;">
            <div class="panel-title" style="color:{info['couleur']}">🛡️ Mesures préventives</div>
        """, unsafe_allow_html=True)
        for prev in info['prevention']:
            st.markdown(f'<div class="prev-item" style="border-left-color:{info["couleur"]};">{prev}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Urgence
        bg_urg = '#fff3cd' if info['gravite'] == 'Modérée' else ('#ffe0de' if info['gravite'] == 'Élevée' else '#e8f8ef')
        st.markdown(f"""
        <div class="urgency" style="background:{bg_urg};border:1px solid {info['border']}40;color:{info['couleur']};">
            <span style="font-size:1.3rem">{info['gravite_icon']}</span>
            <span>{info['urgence']}</span>
        </div>
        """, unsafe_allow_html=True)

# ── Guide Section ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">📚 Guide des Maladies de l\'Olivier</div>', unsafe_allow_html=True)

guide_cols = st.columns(3, gap="large")
for i, (cls, info) in enumerate(CLASS_INFO.items()):
    with guide_cols[i]:
        st.markdown(f"""
        <div class="guide-card" style="background:{info['bg']};border-color:{info['border']}60;">
            <div class="guide-icon">{info['emoji']}</div>
            <div class="guide-title" style="color:{info['couleur']}">{info['nom']}</div>
            <div style="font-style:italic;font-size:0.78rem;opacity:0.5;margin-bottom:0.8rem">{info['nom_sci']}</div>
            <div class="guide-text" style="color:#444">{info['description'][:120]}...</div>
            <div style="margin-top:1rem">
                <span style="background:{info['couleur']}18;color:{info['couleur']};
                             padding:0.3rem 0.8rem;border-radius:20px;font-size:0.78rem;font-weight:600;">
                    {info['gravite_icon']} {info['gravite']}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🫒 <strong>OliveGuard</strong> · Détection des Maladies de l'Olivier par Deep Learning ·
    EfficientNetB4 · Accuracy 90.48% · Encadrant : M. Abdallah Khemais
    <br><small>Projet Deep Learning · 2025-2026</small>
</div>
""", unsafe_allow_html=True)