import streamlit as st
import pandas as pd
import os
import uuid
from datetime import datetime
import requests
from pathlib import Path
import logging
import re
import json

# --- Configuração Inicial ---

# Configuração do logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuração da página
st.set_page_config(
    page_title="Chat Pro",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Constantes e Gerenciamento de Modelos ---
CONVERSATIONS_DIR = Path("conversations")
CONVERSATIONS_DIR.mkdir(exist_ok=True)
MODELS_FILE = Path("models_config.json")

def get_initial_models():
    """Retorna uma estrutura de modelos padrão se o arquivo não existir."""
    # Esta função agora retorna uma lista de dicionários
    initial_free = [
        {'id': 'deepseek/deepseek-chat-v3-0324:free', 'name': 'Deepseek: Deepseek Chat V3 0324 (Free)', 'company': 'Deepseek'},
        {'id': 'google/gemini-2.0-flash-exp:free', 'name': 'Google: Gemini 2.0 Flash Exp (Free)', 'company': 'Google'}
    ]
    initial_paid = [
        {'id': 'anthropic/claude-3.7-sonnet', 'name': 'Anthropic: Claude 3.7 Sonnet', 'company': 'Anthropic'},
        {'id': 'openai/gpt-4o', 'name': 'OpenAI: GPT-4o', 'company': 'OpenAI'}
    ]
    return initial_free, initial_paid

def save_models_to_file(free_models: list, paid_models: list):
    """Salva as listas de dicionários de modelos em um arquivo JSON."""
    try:
        with open(MODELS_FILE, 'w', encoding='utf-8') as f:
            json.dump({'free_models': free_models, 'paid_models': paid_models}, f, indent=4)
        logging.info(f"Modelos salvos em {MODELS_FILE}")
    except Exception as e:
        logging.error(f"Falha ao salvar o arquivo de modelos: {e}")
        st.error(f"Não foi possível salvar as alterações dos modelos: {e}")

def load_models_from_file() -> tuple[list, list]:
    """Carrega as listas de dicionários de modelos do arquivo JSON."""
    if not MODELS_FILE.exists():
        logging.info(f"Arquivo {MODELS_FILE} não encontrado. Criando com modelos padrão.")
        initial_free, initial_paid = get_initial_models()
        save_models_to_file(initial_free, initial_paid)
        return initial_free, initial_paid
    
    try:
        with open(MODELS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Garante que os campos essenciais existam
            free_models = [m for m in data.get('free_models', []) if 'id' in m and 'name' in m and 'company' in m]
            paid_models = [m for m in data.get('paid_models', []) if 'id' in m and 'name' in m and 'company' in m]
            logging.info(f"Modelos carregados de {MODELS_FILE}")
            return free_models, paid_models
    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"Erro ao ler {MODELS_FILE}: {e}. Recriando com modelos padrão.")
        st.warning(f"Arquivo de configuração de modelos corrompido. Restaurando para o padrão.")
        initial_free, initial_paid = get_initial_models()
        save_models_to_file(initial_free, initial_paid)
        return initial_free, initial_paid

# --- FUNÇÃO ATUALIZADA PARA BUSCAR, FILTRAR E ORDENAR MODELOS DA API ---
def fetch_and_classify_models() -> tuple[list | None, list | None]:
    """
    Busca modelos da API, extrai ID, nome e empresa, filtra por data e retorna
    listas ordenadas de dicionários.
    """
    url = "https://openrouter.ai/api/v1/models"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao buscar modelos: {e}")
        logging.error(f"Erro na requisição à API OpenRouter: {e}")
        return None, None

    one_year_ago = datetime.now() - pd.Timedelta(days=365)
    free_models_data = []
    paid_models_data = []

    for model in data.get('data', []):
        model_id = model.get('id')
        model_name = model.get('name')
        created_timestamp = model.get('created', 0)

        if not all([model_id, model_name, created_timestamp]):
            continue
        
        created_date = datetime.fromtimestamp(created_timestamp)
        if created_date < one_year_ago:
            continue

        # Extrai o nome da empresa do campo 'name'
        company_name = model_name.split(':')[0].strip() if ':' in model_name else "Outros"

        model_info = {
            'id': model_id,
            'name': model_name,
            'company': company_name,
            'created': created_timestamp
        }

        if model.get('pricing', {}).get('prompt') == '0':
            free_models_data.append(model_info)
        else:
            paid_models_data.append(model_info)

    free_models_data.sort(key=lambda item: item['created'], reverse=True)
    paid_models_data.sort(key=lambda item: item['created'], reverse=True)
    
    logging.info(f"Modelos buscados: {len(free_models_data)} gratuitos, {len(paid_models_data)} pagos.")
    return free_models_data, paid_models_data

def ordenar_empresas(empresas: list[str]) -> list[str]:
    """
    Ordena a lista de empresas com base na relevância em IA:
    - As 10 mais relevantes aparecem primeiro em ordem de importância
    - As demais são listadas em ordem alfabética
    """
    principais_em_ia = [
        "Google",
        "OpenAI",
	"xAI",
        "Anthropic",
	"MoonshotAI",
	"DeepSeek",
        "Qwen",
        "Perplexity",
        "Meta",
        "Bytedance",
        "MiniMax",
        "Mistral"
    ]

    # Remove duplicatas preservando a ordem de entrada
    empresas = list(dict.fromkeys(empresas))

    # Separa principais e demais
    top = [e for e in principais_em_ia if e in empresas]
    resto = sorted([e for e in empresas if e not in principais_em_ia])
    return top + resto


# Carrega os modelos do arquivo ao iniciar
FREE_MODELS, PAID_MODELS = load_models_from_file()
ALL_MODELS = FREE_MODELS + PAID_MODELS

# Mapeamentos baseados na nova estrutura de dados
MODEL_NAME_MAP = {model['id']: model['name'] for model in ALL_MODELS}
MODEL_ID_MAP = {model['name']: model['id'] for model in ALL_MODELS}
ALL_COMPANIES = ordenar_empresas(list(set(model['company'] for model in ALL_MODELS)))



# Define um modelo padrão
DEFAULT_MODEL_ID = "moonshotai/kimi-k2:free"
if not any(m['id'] == DEFAULT_MODEL_ID for m in ALL_MODELS):
    DEFAULT_MODEL_ID = ALL_MODELS[0]['id'] if ALL_MODELS else None

# Encontra a empresa do modelo padrão
DEFAULT_MODEL_COMPANY = None
if DEFAULT_MODEL_ID:
    default_model_info = next((m for m in ALL_MODELS if m['id'] == DEFAULT_MODEL_ID), None)
    if default_model_info:
        DEFAULT_MODEL_COMPANY = default_model_info['company']

def filter_models(show_free: bool, show_paid: bool, selected_companies: list) -> list:
    """Filtra a lista de dicionários de modelos com base nas preferências."""
    filtered_list = []
    # Garante que a lista de empresas selecionadas seja válida
    if not selected_companies:
        return []
        
    if show_free:
        filtered_list.extend(m for m in FREE_MODELS if m['company'] in selected_companies)
    if show_paid:
        filtered_list.extend(m for m in PAID_MODELS if m['company'] in selected_companies)
    return filtered_list

# --- Estilos CSS ---
st.markdown("""
<style>
/* Estilo geral */
.main {
    background-color: #f0f2f5;
    font-family: 'Inter', sans-serif;
}

/* Estilo personalizado para o botão de consulta web */
.web-search-button {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 14px;
    margin-right: 10px;
}

.web-search-button:hover {
    background-color: #45a049;
}

.web-search-button.active {
    background-color: #2196F3;
}

.web-search-button.active:hover {
    background-color: #1976D2;
}

/* Estilo para container do input */
.input-container {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# --- Funções Auxiliares (sem alterações) ---

def initialize_session_state():
    """Inicializa as variáveis necessárias no estado da sessão."""
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'selected_model_id' not in st.session_state:
        st.session_state.selected_model_id = DEFAULT_MODEL_ID
    if 'api_key' not in st.session_state:
        st.session_state.api_key = "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"
    if 'confirm_delete_id' not in st.session_state:
        st.session_state.confirm_delete_id = None
    if 'show_free_models' not in st.session_state:
        st.session_state.show_free_models = True
    if 'show_paid_models' not in st.session_state:
        st.session_state.show_paid_models = True
    # ATUALIZADO: Inicializa com a empresa padrão
    if 'selected_companies' not in st.session_state:
        st.session_state.selected_companies = [DEFAULT_MODEL_COMPANY] if DEFAULT_MODEL_COMPANY else []
    # Nova variável para o botão de consulta web
    if 'web_search_enabled' not in st.session_state:
        st.session_state.web_search_enabled = False

def call_openrouter_api(model_id: str, api_key: str, conversation_history: list, web_search_enabled: bool = False) -> str:
    """Chama a API OpenRouter Chat Completions de forma segura."""
    if not api_key:
        logging.error("API Key não fornecida para call_openrouter_api.")
        return "Erro: Chave API não configurada."
    if not model_id:
        logging.error("Modelo não fornecido para call_openrouter_api.")
        return "Erro: Modelo não selecionado."
    if not conversation_history:
        logging.warning("Histórico de conversa vazio para call_openrouter_api.")
        return "Erro: Histórico de conversa vazio."

    # Modifica o model_id se a consulta web estiver habilitada
    effective_model_id = f"{model_id}:online" if web_search_enabled else model_id

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Streamlit Chat Pro"
    }
    payload = {
        "model": effective_model_id,
        "messages": conversation_history,
        "max_tokens": 40000,
        "temperature": 0.5,
        "top_p": 0.9,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()

        if (choices := result.get("choices")) and isinstance(choices, list) and len(choices) > 0:
            if (message := choices[0].get("message")) and isinstance(message, dict):
                if (content := message.get("content")) is not None:
                    logging.info(f"API call para modelo {effective_model_id} bem-sucedida.")
                    return str(content)

        logging.error(f"Resposta inesperada da API (modelo {effective_model_id}): {result}")
        return "Erro: Resposta da API em formato inesperado."

    except requests.exceptions.Timeout:
        logging.error(f"Timeout ao chamar a API (modelo {effective_model_id}).")
        return "Erro: A requisição demorou muito para responder (timeout)."
    except requests.exceptions.HTTPError as e:
        error_msg = f"Erro HTTP {response.status_code}: {response.text}"
        logging.error(f"{error_msg} (modelo {effective_model_id}) - Payload: {payload}")
        try:
            error_details = response.json().get("error", {}).get("message", response.text)
            return f"Erro HTTP {response.status_code}: {error_details}"
        except Exception:
            return f"Erro HTTP {response.status_code}: {response.text}"
    except requests.exceptions.RequestException as e:
        logging.error(f"Erro de rede ou conexão ao chamar a API (modelo {effective_model_id}): {e}")
        return f"Erro de conexão: Verifique sua rede ({e})."
    except Exception as e:
        logging.exception(f"Erro inesperado ao chamar a API (modelo {effective_model_id}): {e}")
        return f"Erro inesperado no processamento da API: {e}"


def save_conversation(conversation_id: str, messages: list, model_id: str) -> str:
    """Salva a conversa atual em um arquivo Parquet, incluindo o ID do modelo usado."""
    if not messages:
        logging.warning("Tentativa de salvar conversa vazia.")
        return ""

    try:
        df = pd.DataFrame(messages)
        df['timestamp'] = datetime.now().isoformat()
        df['conversation_id'] = conversation_id
        df['model_id'] = model_id

        filename = CONVERSATIONS_DIR / f"{conversation_id}.parquet"
        df.to_parquet(filename, index=False)
        logging.info(f"Conversa {conversation_id} salva em {filename}")
        return str(filename)
    except Exception as e:
        logging.exception(f"Erro ao salvar a conversa {conversation_id}: {e}")
        st.error(f"Erro ao salvar a conversa: {e}")
        return ""

def load_conversations_metadata() -> list:
    """Carrega metadados das conversas salvas para exibição na sidebar."""
    conversations = []
    for file in CONVERSATIONS_DIR.glob('*.parquet'):
        conversation_id = file.stem
        try:
            df = pd.read_parquet(file, columns=['content', 'timestamp'])
            if not df.empty:
                timestamp_iso = df['timestamp'].iloc[-1] if 'timestamp' in df.columns and not df['timestamp'].empty else None
                if timestamp_iso:
                    try:
                        timestamp_dt = datetime.fromisoformat(timestamp_iso)
                        timestamp_str = timestamp_dt.strftime("%d/%m/%Y %H:%M")
                    except ValueError:
                        logging.warning(f"Timestamp inválido '{timestamp_iso}' no arquivo {file.name}. Usando data de modificação.")
                        timestamp_dt = datetime.fromtimestamp(file.stat().st_mtime)
                        timestamp_str = f"{timestamp_dt.strftime('%d/%m/%Y %H:%M')} (modificado)"
                else:
                    timestamp_dt = datetime.fromtimestamp(file.stat().st_mtime)
                    timestamp_str = f"{timestamp_dt.strftime('%d/%m/%Y %H:%M')} (modificado)"

                first_message_content = df['content'].iloc[0] if 'content' in df.columns and not df['content'].empty else "Conversa iniciada"
                preview = first_message_content[:40] + "..." if len(first_message_content) > 40 else first_message_content

                conversations.append({
                    "id": conversation_id,
                    "preview": preview,
                    "timestamp_str": timestamp_str,
                    "timestamp_dt": timestamp_dt
                })
            else:
                logging.warning(f"Arquivo de conversa vazio encontrado: {file.name}. Ignorando.")
        except Exception as e:
            logging.error(f"Erro ao carregar metadados da conversa {file.name}: {e}")
            conversations.append({
                "id": conversation_id,
                "preview": f"Erro ao carregar ({e})",
                "timestamp_str": "Inválido",
                "timestamp_dt": datetime.min,
                "error": True
            })

    conversations.sort(key=lambda x: x.get('timestamp_dt', datetime.min) if not x.get('error') else datetime.min, reverse=True)
    return conversations

def load_conversation_messages(conversation_id: str) -> tuple[list, str | None]:
    """Carrega as mensagens e o ID do modelo de uma conversa específica."""
    filename = CONVERSATIONS_DIR / f"{conversation_id}.parquet"
    if not filename.exists():
        st.error(f"Arquivo da conversa {conversation_id} não encontrado.")
        return [], None

    try:
        df = pd.read_parquet(filename, columns=['role', 'content', 'model_id'])
        messages = df[['role', 'content']].to_dict('records')
        model_id = df['model_id'].iloc[0] if not df.empty and 'model_id' in df.columns else None
        logging.info(f"Conversa {conversation_id} carregada com {len(messages)} mensagens. Modelo ID: {model_id}")
        return messages, model_id
    except Exception as e:
        logging.exception(f"Erro ao ler o arquivo da conversa {filename.name}: {e}")
        st.error(f"Erro ao ler o arquivo da conversa {filename.name}: {e}")
        return [], None

def delete_conversation(conversation_id: str):
    """Exclui o arquivo de uma conversa específica."""
    filename = CONVERSATIONS_DIR / f"{conversation_id}.parquet"
    try:
        if filename.exists():
            filename.unlink()
            logging.info(f"Conversa {conversation_id} excluída com sucesso.")
            st.toast(f"Conversa '{conversation_id[:8]}...' excluída.", icon="🗑️")
            st.session_state.confirm_delete_id = None
            if st.session_state.conversation_id == conversation_id:
                st.session_state.messages = []
                st.session_state.conversation_id = str(uuid.uuid4())
                # Lógica para resetar o modelo
                available_models = filter_models(st.session_state.show_free_models, st.session_state.show_paid_models, st.session_state.selected_companies)
                if any(m['id'] == DEFAULT_MODEL_ID for m in available_models):
                    st.session_state.selected_model_id = DEFAULT_MODEL_ID
                elif available_models:
                    st.session_state.selected_model_id = available_models[0]['id']
                else:
                    st.session_state.selected_model_id = None
                logging.info("Conversa ativa foi excluída. Iniciando nova conversa.")
        else:
            logging.warning(f"Tentativa de excluir conversa inexistente: {conversation_id}")
    except Exception as e:
        logging.exception(f"Erro ao excluir a conversa {conversation_id}: {e}")
        st.error(f"Erro ao excluir a conversa: {e}")
    finally:
        st.session_state.confirm_delete_id = None

# --- Inicialização da Sessão ---
initialize_session_state()

# --- Barra Lateral (Sidebar) ---
with st.sidebar:
    st.header("🤖 Configurações")

    if st.button("Nova Conversa", key="new_chat_button"):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.confirm_delete_id = None
        st.toast("Nova conversa iniciada!", icon="🤖")
        st.rerun()

    st.divider()

    # NOVO CÓDIGO DO INTERRUPTOR DE BUSCA WEB
    st.subheader("🌐 Consulta Web")

    # Garante que o estado existe na primeira execução
    if 'web_search_enabled' not in st.session_state:
        st.session_state.web_search_enabled = False

    # O toggle agora lê seu valor do estado, mas não o possui (sem 'key')
    # e então atualizamos o estado com a interação do usuário.
    st.session_state.web_search_enabled = st.toggle(
    "Ativar busca na web para a próxima mensagem",
    value=st.session_state.web_search_enabled,
    help="Se ativado, o modelo de IA terá acesso à internet para formular a resposta."
)

    st.divider()

    # --- Filtros de Modelo ---
    st.subheader("Filtros de Modelos")

    # Filtro Gratuito/Pago (vem primeiro para determinar as empresas)
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.checkbox("Gratuitos", key="show_free_models")
    with col_f2:
        st.checkbox("Pagos", key="show_paid_models")

    # Determinar empresas disponíveis com base no filtro de custo
    temp_models_for_company_filter = []
    if st.session_state.show_free_models:
        temp_models_for_company_filter.extend(FREE_MODELS)
    if st.session_state.show_paid_models:
        temp_models_for_company_filter.extend(PAID_MODELS)
    

    #available_companies = sorted(list(set(m['company'] for m in temp_models_for_company_filter)))
    available_companies = ordenar_empresas(m['company'] for m in temp_models_for_company_filter)

    # Limpar a seleção de empresas se elas não estiverem mais disponíveis
    st.session_state.selected_companies = [
        c for c in st.session_state.selected_companies if c in available_companies
    ]

    # Filtro por Empresa (agora com opções dinâmicas)
    st.multiselect(
        "Empresas",
        options=available_companies,
        key="selected_companies",
        help="Selecione uma ou mais empresas para filtrar os modelos."
    )

    # --- Caixa de Seleção de Modelos ---
    available_models = filter_models(
        st.session_state.show_free_models,
        st.session_state.show_paid_models,
        st.session_state.selected_companies
    )
    available_model_names = [m['name'] for m in available_models]
    
    if not available_models:
        st.warning("Nenhum modelo disponível com os filtros selecionados.")
    else:
        current_model_name = MODEL_NAME_MAP.get(st.session_state.selected_model_id)
        current_index = 0
        # Verifica se o modelo atual ainda está na lista filtrada
        if current_model_name in available_model_names:
            current_index = available_model_names.index(current_model_name)
        else:
            # Se o modelo atual não está na lista, seleciona o primeiro disponível
            st.session_state.selected_model_id = available_models[0]['id']
            logging.info(f"Modelo anterior filtrado. Selecionando o primeiro disponível: {st.session_state.selected_model_id}")
            # O rerun() aqui garante que o selectbox será atualizado com o novo índice
            st.rerun()

        selected_friendly_name = st.selectbox(
            "Modelo de IA",
            options=available_model_names,
            index=current_index,
            key="model_selector",
            help="Escolha o modelo de linguagem para interagir."
        )
        if selected_friendly_name and MODEL_ID_MAP.get(selected_friendly_name) != st.session_state.selected_model_id:
            st.session_state.selected_model_id = MODEL_ID_MAP[selected_friendly_name]
            logging.info(f"Modelo selecionado alterado para: {st.session_state.selected_model_id}")
            st.rerun()


    st.divider() # Mais um divisor

    # Conversas Salvas
    st.header("📂 Conversas Salvas")
    conversations = load_conversations_metadata()

    if not conversations:
        st.caption("Nenhuma conversa salva encontrada.")
    else:
        for conv in conversations:
            conv_id = conv['id']
            is_confirming_delete = (st.session_state.confirm_delete_id == conv_id)
            item_container = st.container()
            with item_container:
                col_load, col_delete = st.columns([0.8, 0.2])
                with col_load:
                    preview_text = conv.get('preview', 'Conversa')
                    if st.button(preview_text, key=f"load_btn_{conv_id}_action", type="primary"):
                        if conv.get("error"):
                            st.error(f"Não é possível carregar a conversa {conv_id} devido a um erro anterior.")
                        else:
                            messages, loaded_model_id = load_conversation_messages(conv_id)
                            if messages is not None:
                                st.session_state.conversation_id = conv_id
                                st.session_state.messages = messages
                                if loaded_model_id and any(m['id'] == loaded_model_id for m in ALL_MODELS):
                                    st.session_state.selected_model_id = loaded_model_id
                                else:
                                    st.warning(f"Modelo '{loaded_model_id}' da conversa não encontrado na lista atual.")
                                
                                st.session_state.confirm_delete_id = None
                                st.toast(f"Conversa '{conv['preview']}' carregada.", icon="📂")
                                logging.info(f"Conversa {conv_id} carregada.")
                                st.rerun()
                with col_delete:
                    if is_confirming_delete:
                        if st.button("✔️", key=f"confirm_delete_{conv_id}", help="Confirmar exclusão"):
                            delete_conversation(conv_id)
                            st.rerun()
                        if st.button("❌", key=f"cancel_delete_{conv_id}", help="Cancelar exclusão"):
                            st.session_state.confirm_delete_id = None
                            st.rerun()
                    else:
                        if st.button("🗑️", key=f"delete_{conv_id}", help="Excluir esta conversa"):
                            st.session_state.confirm_delete_id = conv_id
                            st.rerun()

    st.divider()

    # --- Seção de Manutenção de Modelos ---
    with st.expander("🔧 Manutenção de Modelos"):
        st.subheader("Gerenciar Listas de Modelos")
        st.caption("Atualize as listas para obter os modelos mais recentes da API.")

        if st.button("🔄 Atualizar Automaticamente da API", type="primary"):
            with st.spinner("Buscando modelos mais recentes..."):
                new_free, new_paid = fetch_and_classify_models()
                if new_free is not None and new_paid is not None:
                    save_models_to_file(new_free, new_paid)
                    st.toast("Listas de modelos atualizadas com sucesso!", icon="✅")
                    st.rerun()
                else:
                    st.toast("Falha ao atualizar os modelos.", icon="❌")

    st.divider()

    # Entrada da Chave API
    st.text_input(
        "🔑 Chave API OpenRouter",
        type="password",
        key="api_key",
        help="Insira sua chave API do OpenRouter."
    )
    if not st.session_state.api_key:
        st.warning("⚠️ Insira sua chave API para usar o chat.")
    st.markdown("[Obter chave OpenRouter](https://openrouter.ai/keys)", unsafe_allow_html=True)


# --- Área Principal do Chat ---
st.title("💬 Chat Pro com OpenRouter")
selected_model_name = MODEL_NAME_MAP.get(st.session_state.selected_model_id, "Nenhum modelo selecionado")
# Adiciona indicação de consulta web no nome do modelo se estiver habilitada
model_display_name = f"{selected_model_name}:online" if st.session_state.web_search_enabled else selected_model_name
st.caption(f"Conversa ID: `{st.session_state.conversation_id[:8]}...` | Modelo: `{model_display_name}`")

if not st.session_state.messages:
    st.info("👋 Olá! Digite sua mensagem abaixo para começar.")
else:
    for message in st.session_state.messages:
        role = message.get('role', 'unknown')
        avatar = "🧑" if role == "user" else "🤖"
        with st.chat_message(name=role, avatar=avatar):
            st.markdown(message.get('content', ''))

# 3. Entrada de Prompt do Usuário (sempre no fundo)
input_disabled = not st.session_state.api_key or not st.session_state.selected_model_id
prompt = st.chat_input(
    "Digite sua mensagem aqui...",
    key="prompt_input",
    disabled=input_disabled,
)

# 4. Lógica de processamento do prompt (idêntica à da Opção 1)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_conversation(st.session_state.conversation_id, st.session_state.messages, st.session_state.selected_model_id)
    
    with st.chat_message(name="user", avatar="🧑"):
        st.markdown(prompt)
        
    with st.spinner("Pensando... 🤔"):
        response_content = call_openrouter_api(
            model_id=st.session_state.selected_model_id,
            api_key=st.session_state.api_key,
            conversation_history=st.session_state.messages,
            web_search_enabled=st.session_state.web_search_enabled
        )
    
    if st.session_state.web_search_enabled:
        st.session_state.web_search_enabled = False

    st.session_state.messages.append({"role": "assistant", "content": response_content})
    save_conversation(st.session_state.conversation_id, st.session_state.messages, st.session_state.selected_model_id)
    st.rerun()

# --- Rodapé ---
st.markdown("---")