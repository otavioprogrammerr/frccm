import cv2
import os
import numpy as np
import sqlite3
from datetime import datetime, time as timeobj
import base64
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user
import sys
import shutil
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from serverless_wsgi import handle_request  # type: ignore

# Carrega variáveis de ambiente
load_dotenv()

# Configuração do Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'rostos')
app.config['DATABASE_URL'] = os.getenv('DATABASE_URL',
                                       'sqlite:///static/registro.db')
app.secret_key = os.getenv('SECRET_KEY', 'sua-chave-secreta-aqui')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Configuração do Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin):

    def __init__(self, id):
        self.id = id


# Usuário único com senha fixa
USUARIO = {
    "id": 1,
    "username": "admin",
    "password": os.getenv('ADMIN_PASSWORD', 'olavo-ccm')
}


@login_manager.user_loader
def load_user(user_id):
    return User(user_id)


# Verificação do módulo OpenCV
try:
    if hasattr(cv2, 'face'):
        face_module = cv2.face
    elif hasattr(cv2, 'face_LBPH'):
        face_module = cv2.face_LBPH
    else:
        from cv2 import face as face_module
except (ImportError, AttributeError) as e:
    raise ImportError(
        "Módulo face não encontrado. Instale opencv-contrib-python-headless"
    ) from e


# Funções auxiliares
def get_db_connection():
    if app.config['DATABASE_URL'].startswith('postgres://'):
        conn = psycopg2.connect(app.config['DATABASE_URL'], sslmode='require')
        conn.autocommit = True
        return conn
    else:
        db_path = app.config['DATABASE_URL'].replace('sqlite:///', '')
        # Garante que o diretório existe
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return sqlite3.connect(db_path)

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    if app.config['DATABASE_URL'].startswith('postgres://'):
        # Comandos para PostgreSQL
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pessoas (
                id SERIAL PRIMARY KEY,
                nome TEXT NOT NULL,
                turno TEXT NOT NULL,
                turma TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS registros (
                id SERIAL PRIMARY KEY,
                pessoa_id INTEGER REFERENCES pessoas(id),
                data TEXT NOT NULL,
                hora TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS turmas (
                id SERIAL PRIMARY KEY,
                nome TEXT NOT NULL UNIQUE
            )
        ''')
        cursor.execute(
            "INSERT INTO turmas (nome) VALUES ('6A') ON CONFLICT (nome) DO NOTHING"
        )
    else:
        # Comandos para SQLite
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pessoas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome TEXT NOT NULL,
                turno TEXT NOT NULL,
                turma TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS registros (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pessoa_id INTEGER,
                data TEXT NOT NULL,
                hora TEXT NOT NULL,
                FOREIGN KEY (pessoa_id) REFERENCES pessoas (id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS turmas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome TEXT NOT NULL UNIQUE
            )
        ''')
        cursor.execute("INSERT OR IGNORE INTO turmas (nome) VALUES ('6A')")

    conn.commit()
    conn.close()


def allowed_file(filename):
    return '.' in filename and filename.rsplit(
        '.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def get_rosto_path(nome):
    return os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(nome))


def processar_rosto(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                         "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.1,
                                          minNeighbors=5)
    if len(faces) == 0:
        return None
    (x, y, w, h) = faces[0]
    return gray[y:y + h, x:x + w]


def criar_reconhecedor():
    try:
        return face_module.LBPHFaceRecognizer_create()
    except AttributeError:
        try:
            return cv2.face_LBPH.LBPHFaceRecognizer_create()
        except AttributeError:
            try:
                return cv2.createLBPHFaceRecognizer()
            except AttributeError as e:
                raise Exception(
                    "Não foi possível criar o reconhecedor facial") from e


def treinar_reconhecedor():
    recognizer = criar_reconhecedor()
    nomes_map = {}
    turnos_map = {}
    turmas_map = {}
    imagens = []
    ids = []

    conn = get_db_connection()
    try:
        if app.config['DATABASE_URL'].startswith('postgres://'):
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT id, nome, turno, turma FROM pessoas")
        else:
            cursor = conn.cursor()
            cursor.execute("SELECT id, nome, turno, turma FROM pessoas")

        pessoas = cursor.fetchall()

        for pessoa in pessoas:
            if app.config['DATABASE_URL'].startswith('postgres://'):
                pessoa_id, nome, turno, turma = pessoa['id'], pessoa[
                    'nome'], pessoa['turno'], pessoa['turma']
            else:
                pessoa_id, nome, turno, turma = pessoa

            pasta = get_rosto_path(nome)
            if not os.path.exists(pasta):
                continue

            for arquivo in os.listdir(pasta):
                if arquivo.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(pasta, arquivo)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        imagens.append(img)
                        ids.append(pessoa_id)

            nomes_map[pessoa_id] = nome
            turnos_map[pessoa_id] = turno
            turmas_map[pessoa_id] = turma

    finally:
        conn.close()

    if len(imagens) == 0:
        return None, None, None, None

    recognizer.train(imagens, np.array(ids))
    return recognizer, nomes_map, turnos_map, turmas_map


# Rotas de autenticação
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username == USUARIO['username'] and password == USUARIO['password']:
            user = User(USUARIO['id'])
            login_user(user)
            return redirect(url_for('index'))

        return render_template('login.html', error="Credenciais inválidas")

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# Rotas principais
@app.route('/')
@login_required
def index():
    return render_template('index.html')


@app.route('/cadastro')
@login_required
def cadastro():
    conn = get_db_connection()
    try:
        if app.config['DATABASE_URL'].startswith('postgres://'):
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT nome FROM turmas")
            turmas = [row['nome'] for row in cursor.fetchall()]
        else:
            cursor = conn.cursor()
            cursor.execute("SELECT nome FROM turmas")
            turmas = [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()
    return render_template('cadastro.html', turmas=turmas)


@app.route('/api/capturar_fotos', methods=['POST'])
@login_required
def api_capturar_fotos():
    data = request.get_json()
    nome = data.get('nome')
    turno = data.get('turno')
    turma = data.get('turma')
    fotos = data.get('fotos', [])

    if not nome or not turno or not turma:
        return jsonify({
            'success': False,
            'message': 'Preencha todos os campos!'
        })

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if app.config['DATABASE_URL'].startswith('postgres://'):
            cursor.execute("SELECT id FROM pessoas WHERE nome = %s", (nome, ))
        else:
            cursor.execute("SELECT id FROM pessoas WHERE nome = ?", (nome, ))

        if cursor.fetchone():
            return jsonify({
                'success': False,
                'message': f'A pessoa "{nome}" já está cadastrada!'
            })

        if app.config['DATABASE_URL'].startswith('postgres://'):
            cursor.execute(
                "INSERT INTO pessoas (nome, turno, turma) VALUES (%s, %s, %s) RETURNING id",
                (nome, turno, turma))
            pessoa_id = cursor.fetchone()['id']
        else:
            cursor.execute(
                "INSERT INTO pessoas (nome, turno, turma) VALUES (?, ?, ?)",
                (nome, turno, turma))
            pessoa_id = cursor.lastrowid

        conn.commit()

    finally:
        conn.close()

    pasta = get_rosto_path(nome)
    os.makedirs(pasta, exist_ok=True)

    for i, foto_data in enumerate(fotos):
        try:
            header, encoded = foto_data.split(",", 1)
            binary_data = base64.b64decode(encoded)
            nparr = np.frombuffer(binary_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is not None:
                rosto = processar_rosto(img)
                if rosto is not None:
                    cv2.imwrite(os.path.join(pasta, f"{i}.jpg"), rosto)
        except Exception as e:
            print(f"Erro ao processar foto {i}: {str(e)}")
            continue

    return jsonify({
        'success': True,
        'message': f'Cadastro de {nome} realizado com sucesso!'
    })


@app.route('/monitorar')
@login_required
def monitorar():
    return render_template('monitorar.html')


@app.route('/api/reconhecer', methods=['POST'])
@login_required
def api_reconhecer():
    try:
        recognizer, nomes_map, turnos_map, turmas_map = treinar_reconhecedor()
        if recognizer is None:
            return jsonify({
                'success':
                False,
                'message':
                'Nenhum rosto cadastrado para reconhecimento'
            })

        data = request.get_json()
        image_data = data['image'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        resultados = []
        for (x, y, w, h) in faces:
            rosto = gray[y:y + h, x:x + w]
            id_, confianca = recognizer.predict(rosto)

            if confianca < 50:
                nome = nomes_map.get(id_, "Desconhecido")
                turno = turnos_map.get(id_, "turno_desconhecido")
                turma = turmas_map.get(id_, "turma_desconhecida")

                if nome == "Desconhecido":
                    continue

                agora = datetime.now()
                if turno == "manha":
                    limite = datetime.combine(agora.date(), timeobj(7, 10))
                    fim = datetime.combine(agora.date(), timeobj(12, 30))
                elif turno == "tarde":
                    limite = datetime.combine(agora.date(), timeobj(13, 10))
                    fim = datetime.combine(agora.date(), timeobj(18, 30))
                else:
                    limite = datetime.combine(agora.date(), timeobj(0, 0))
                    fim = datetime.combine(agora.date(), timeobj(23, 59))

                if limite <= agora <= fim:
                    data_str = agora.strftime("%d-%m-%Y")
                    hora_str = agora.strftime("%H:%M:%S")

                    conn = get_db_connection()
                    try:
                        cursor = conn.cursor()
                        if app.config['DATABASE_URL'].startswith(
                                'postgres://'):
                            cursor.execute(
                                '''
                                SELECT id FROM registros 
                                WHERE pessoa_id = %s AND data = %s
                            ''', (id_, data_str))
                        else:
                            cursor.execute(
                                '''
                                SELECT id FROM registros 
                                WHERE pessoa_id = ? AND data = ?
                            ''', (id_, data_str))

                        if not cursor.fetchone():
                            if app.config['DATABASE_URL'].startswith(
                                    'postgres://'):
                                cursor.execute(
                                    '''
                                    INSERT INTO registros (pessoa_id, data, hora)
                                    VALUES (%s, %s, %s)
                                ''', (id_, data_str, hora_str))
                            else:
                                cursor.execute(
                                    '''
                                    INSERT INTO registros (pessoa_id, data, hora)
                                    VALUES (?, ?, ?)
                                ''', (id_, data_str, hora_str))
                            conn.commit()
                            registrado = True
                        else:
                            registrado = False
                    finally:
                        conn.close()

                    resultados.append({
                        'nome': nome,
                        'confianca': float(confianca),
                        'registrado': registrado,
                        'turno': turno,
                        'turma': turma
                    })

        return jsonify({
            'success': True,
            'resultados': resultados,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Erro no reconhecimento: {str(e)}'
        })


@app.route('/relatorios')
@login_required
def relatorios():
    conn = get_db_connection()
    try:
        if app.config['DATABASE_URL'].startswith('postgres://'):
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT nome FROM turmas")
            turmas = [row['nome'] for row in cursor.fetchall()]
        else:
            cursor = conn.cursor()
            cursor.execute("SELECT nome FROM turmas")
            turmas = [row[0] for row in cursor.fetchall()]

        dados_por_turma = {}
        for turma in turmas:
            if app.config['DATABASE_URL'].startswith('postgres://'):
                cursor.execute(
                    '''
                    SELECT p.nome, p.turno, COUNT(r.id) as total_atrasos, 
                           SUM(CASE 
                               WHEN p.turno = 'manha' AND r.hora > '07:10:00' THEN 
                                   EXTRACT(EPOCH FROM (r.hora::time - '07:10:00'::time))
                               WHEN p.turno = 'tarde' AND r.hora > '13:10:00' THEN 
                                   EXTRACT(EPOCH FROM (r.hora::time - '13:10:00'::time))
                               ELSE 0
                           END) as total_atraso
                    FROM pessoas p
                    LEFT JOIN registros r ON p.id = r.pessoa_id
                    WHERE p.turma = %s
                    GROUP BY p.id, p.nome, p.turno
                ''', (turma, ))

                dados_por_turma[turma] = []
                for row in cursor.fetchall():
                    total_segundos_atraso = row['total_atraso'] or 0
                    minutos = int(total_segundos_atraso // 60)
                    segundos = int(total_segundos_atraso % 60)
                    dados_por_turma[turma].append({
                        'nome':
                        row['nome'],
                        'turno':
                        row['turno'],
                        'total_atrasos':
                        row['total_atrasos'],
                        'tempo_atraso':
                        f"{minutos} min {segundos} s"
                    })
            else:
                cursor.execute(
                    '''
                    SELECT p.nome, p.turno, COUNT(r.id) as total_atrasos, 
                           SUM(CASE 
                               WHEN p.turno = 'manha' AND TIME(r.hora) > TIME('07:10:00') THEN 
                                   (JULIANDAY(TIME(r.hora)) - JULIANDAY(TIME('07:10:00'))) * 86400
                               WHEN p.turno = 'tarde' AND TIME(r.hora) > TIME('13:10:00') THEN 
                                   (JULIANDAY(TIME(r.hora)) - JULIANDAY(TIME('13:10:00'))) * 86400
                               ELSE 0
                           END) as total_atraso
                    FROM pessoas p
                    LEFT JOIN registros r ON p.id = r.pessoa_id
                    WHERE p.turma = ?
                    GROUP BY p.id
                ''', (turma, ))

                dados_por_turma[turma] = []
                for row in cursor.fetchall():
                    total_segundos_atraso = row[3] or 0
                    minutos = int(total_segundos_atraso // 60)
                    segundos = int(total_segundos_atraso % 60)
                    dados_por_turma[turma].append({
                        'nome':
                        row[0],
                        'turno':
                        row[1],
                        'total_atrasos':
                        row[2],
                        'tempo_atraso':
                        f"{minutos} min {segundos} s"
                    })
    finally:
        conn.close()

    return render_template('relatorios.html',
                           turmas=turmas,
                           dados_por_turma=dados_por_turma)


@app.route('/gerenciar_alunos')
@login_required
def gerenciar_alunos():
    conn = get_db_connection()
    try:
        if app.config['DATABASE_URL'].startswith('postgres://'):
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT p.id, p.nome, p.turno, p.turma, COUNT(r.id) as total_registros
                FROM pessoas p
                LEFT JOIN registros r ON p.id = r.pessoa_id
                GROUP BY p.id, p.nome, p.turno, p.turma
                ORDER BY p.nome
            """)
            alunos = cursor.fetchall()

            cursor.execute("SELECT nome FROM turmas")
            turmas = [row['nome'] for row in cursor.fetchall()]
        else:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT p.id, p.nome, p.turno, p.turma, COUNT(r.id) as total_registros
                FROM pessoas p
                LEFT JOIN registros r ON p.id = r.pessoa_id
                GROUP BY p.id
                ORDER BY p.nome
            """)
            alunos = cursor.fetchall()

            cursor.execute("SELECT nome FROM turmas")
            turmas = [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()

    return render_template('gerenciar_alunos.html',
                           alunos=alunos,
                           turmas=turmas)


@app.route('/api/registros_aluno/<int:aluno_id>')
@login_required
def api_registros_aluno(aluno_id):
    conn = get_db_connection()
    try:
        if app.config['DATABASE_URL'].startswith('postgres://'):
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT r.data, r.hora, p.turno
                FROM registros r
                JOIN pessoas p ON r.pessoa_id = p.id
                WHERE r.pessoa_id = %s
                ORDER BY r.data DESC, r.hora DESC
            """, (aluno_id,))
            registros = cursor.fetchall()
            
            cursor.execute("SELECT nome FROM pessoas WHERE id = %s", (aluno_id,))
            nome_aluno = cursor.fetchone()['nome']
        else:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT r.data, r.hora, p.turno
                FROM registros r
                JOIN pessoas p ON r.pessoa_id = p.id
                WHERE r.pessoa_id = ?
                ORDER BY r.data DESC, r.hora DESC
            """, (aluno_id,))
            registros = cursor.fetchall()
            
            cursor.execute("SELECT nome FROM pessoas WHERE id = ?", (aluno_id,))
            nome_aluno = cursor.fetchone()[0]

        # Filtrar apenas os registros que são atrasados
        atrasos = []
        for registro in registros:
            if isinstance(registro, dict):
                hora_str = registro['hora']
                turno = registro['turno']
            else:
                hora_str = registro[1]
                turno = registro[2]
            
            hora_registro = datetime.strptime(hora_str, "%H:%M:%S").time()
            
            if turno == "manha" and hora_registro > datetime.strptime("07:10:00", "%H:%M:%S").time():
                atrasos.append({
                    'data': registro['data'] if isinstance(registro, dict) else registro[0],
                    'hora': hora_str,
                    'status': 'Atraso'
                })
            elif turno == "tarde" and hora_registro > datetime.strptime("13:10:00", "%H:%M:%S").time():
                atrasos.append({
                    'data': registro['data'] if isinstance(registro, dict) else registro[0],
                    'hora': hora_str,
                    'status': 'Atraso'
                })

        return jsonify({
            'success': True,
            'nome_aluno': nome_aluno,
            'atrasos': atrasos
        })
    finally:
        conn.close()


@app.route('/api/deletar_aluno', methods=['POST'])
@login_required
def api_deletar_aluno():
    try:
        data = request.get_json()
        aluno_id = data.get('id')

        if not aluno_id:
            return jsonify({
                'success': False,
                'message': 'ID do aluno não fornecido'
            })

        conn = get_db_connection()
        try:
            cursor = conn.cursor()

            if app.config['DATABASE_URL'].startswith('postgres://'):
                cursor.execute("SELECT nome FROM pessoas WHERE id = %s",
                               (aluno_id, ))
            else:
                cursor.execute("SELECT nome FROM pessoas WHERE id = ?",
                               (aluno_id, ))

            aluno = cursor.fetchone()

            if not aluno:
                return jsonify({
                    'success': False,
                    'message': 'Aluno não encontrado'
                })

            nome_aluno = aluno[0] if isinstance(aluno, dict) else aluno[0]
            pasta_aluno = get_rosto_path(nome_aluno)

            if app.config['DATABASE_URL'].startswith('postgres://'):
                cursor.execute("DELETE FROM registros WHERE pessoa_id = %s",
                               (aluno_id, ))
                cursor.execute("DELETE FROM pessoas WHERE id = %s",
                               (aluno_id, ))
            else:
                cursor.execute("DELETE FROM registros WHERE pessoa_id = ?",
                               (aluno_id, ))
                cursor.execute("DELETE FROM pessoas WHERE id = ?",
                               (aluno_id, ))

            conn.commit()

            if os.path.exists(pasta_aluno):
                shutil.rmtree(pasta_aluno)

            return jsonify({
                'success':
                True,
                'message':
                f'Aluno {nome_aluno} deletado com sucesso!'
            })
        finally:
            conn.close()

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Erro ao deletar aluno: {str(e)}'
        })


@app.route('/api/turmas', methods=['GET', 'POST', 'DELETE'])
@login_required
def api_turmas():
    if request.method == 'POST':
        nova_turma = request.json.get('nome')
        if not nova_turma:
            return jsonify({
                'success': False,
                'message': 'Nome da turma é obrigatório'
            })

        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            if app.config['DATABASE_URL'].startswith('postgres://'):
                cursor.execute("INSERT INTO turmas (nome) VALUES (%s)",
                               (nova_turma, ))
            else:
                cursor.execute("INSERT INTO turmas (nome) VALUES (?)",
                               (nova_turma, ))
            conn.commit()
            return jsonify({
                'success': True,
                'message': f'Turma {nova_turma} adicionada'
            })
        except (sqlite3.IntegrityError, psycopg2.IntegrityError):
            return jsonify({
                'success': False,
                'message': f'Turma {nova_turma} já existe'
            })
        finally:
            conn.close()

    elif request.method == 'DELETE':
        turma = request.json.get('nome')
        if not turma:
            return jsonify({
                'success': False,
                'message': 'Nome da turma é obrigatório'
            })

        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            if app.config['DATABASE_URL'].startswith('postgres://'):
                cursor.execute("DELETE FROM turmas WHERE nome = %s", (turma, ))
            else:
                cursor.execute("DELETE FROM turmas WHERE nome = ?", (turma, ))
            conn.commit()
            return jsonify({
                'success': True,
                'message': f'Turma {turma} removida'
            })
        finally:
            conn.close()

    else:
        conn = get_db_connection()
        try:
            if app.config['DATABASE_URL'].startswith('postgres://'):
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute("SELECT nome FROM turmas")
                turmas = [row['nome'] for row in cursor.fetchall()]
            else:
                cursor = conn.cursor()
                cursor.execute("SELECT nome FROM turmas")
                turmas = [row[0] for row in cursor.fetchall()]
            return jsonify({'success': True, 'turmas': turmas})
        finally:
            conn.close()


@app.route('/api/limpar_registros', methods=['POST'])
@login_required
def api_limpar_registros():
    senha = request.json.get('senha')
    if senha != "olavo":
        return jsonify({'success': False, 'message': 'Senha incorreta'})

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM registros")
        conn.commit()
        return jsonify({
            'success': True,
            'message': 'Todos os registros foram apagados'
        })
    finally:
        conn.close()


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


def vercel_handler(request, context):
    return handle_request(app, request, context)


if __name__ == '__main__':
    init_db()
    app.run(debug=True)