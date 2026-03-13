
import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List


#Dependencies
try:
    import whisper
except ImportError:
    sys.exit("Instale o whisper: pip install openai-whisper")

try:
    import ffmpeg
except ImportError:
    sys.exit("Instale o ffmpeg-python: pip install ffmpeg-python")

try:
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize
except ImportError:
    sys.exit("Instale o nltk: pip install nltk")



#Estrtura 
@dataclass
class Sentenca:
    id: int 
    texto: str
    inicio: float
    fim: float 
    inicio_fmt: str
    fim_fmt: str
    duracao: float
    num_palavras: int 
    num_caracteres: int
    confianca_media: float

#Utils

def segundos_para_hhmss(seg: float) -> str: 
    seg = max(0.0, seg)
    h = int (seg // 3600)
    m = int((seg % 3600) // 60)
    s = seg % 60 
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def extrair_audio(caminho_video: str, caminho_audio: str = "audio_temp.wav") -> str:
    print(f"Extraindo o audui de: {caminho_video} ")
    (
    ffmpeg
    .input(caminho_video)
    .output(
        caminho_audio,
        format="wav",
        acodec="pcm_s16le",
        ac=1,           # mono
        ar=16000,       # 16 kHz
    )
    .overwrite_output()
    .run(quiet=True)
    )
    print(f"Audio salvo em: {caminho_audio}")
    return caminho_audio

def transcrever_audio(caminho_audio: str, modelo: str = "base", idioma: str = None):
    print(f"Carregando modelo Whisper '{modelo}'…")
    model = whisper.load_model(modelo)

    opcoes = dict(word_timestamps=True)
    if idioma:
        opcoes["language"] = idioma

    print("Transcrevendo… (pode levar alguns minutos)")
    resultado = model.transcribe(caminho_audio, **opcoes)
    print(f"Idioma detectado: {resultado.get('language', 'desconhecido')}")
    return resultado

#Lógica principal 

def segmentos_para_palavras(resultado_whisper) -> List[dict]:
    palavras = []
    for seg in resultado_whisper["segments"]:
        for w in seg.get("words", []):
            palavras.append({
                "palavra": w["word"].strip(),
                "inicio": w["start"],
                "fim": w["end"],
                "probabilidade": w.get("probability", 1.0),
            })
    return palavras

def texto_completo_de_palavras(palavras: List[dict]) -> str:
    return " ".join(p["palavra"] for p in palavras)

def dividir_sentenca_longa(sentenca: Sentenca, palavras: List[dict], max_duracao: float = 30.0) -> List[Sentenca]:
    """Divide sentenças muito longas em partes menores por tempo máximo."""
    if sentenca.duracao <= max_duracao:
        return [sentenca]

    palavras_sent = [p for p in palavras if sentenca.inicio <= p["inicio"] <= sentenca.fim]
    
    partes = []
    parte_atual = []
    inicio_parte = None

    for pw in palavras_sent:
        if inicio_parte is None:
            inicio_parte = pw["inicio"]
        
        parte_atual.append(pw)
        duracao_atual = pw["fim"] - inicio_parte

        if duracao_atual >= max_duracao:
            partes.append((inicio_parte, pw["fim"], parte_atual))
            parte_atual = []
            inicio_parte = None

    if parte_atual:
        partes.append((inicio_parte, parte_atual[-1]["fim"], parte_atual))

    
    resultado = []
    for i, (inicio, fim, pals) in enumerate(partes):
        texto = " ".join(p["palavra"] for p in pals)
        probs = [p["probabilidade"] for p in pals]
        resultado.append(Sentenca(
            id=sentenca.id,
            texto=texto.strip(),
            inicio=round(inicio, 3),
            fim=round(fim, 3),
            inicio_fmt=segundos_para_hhmss(inicio),
            fim_fmt=segundos_para_hhmss(fim),
            duracao=round(fim - inicio, 3),
            num_palavras=len(pals),
            num_caracteres=len(texto.strip()),
            confianca_media=round(sum(probs) / len(probs), 4),
        ))
    return resultado


def dividir_em_sentencas(palavras: List[dict], idioma: str = "portuguese") -> List[Sentenca]:
    texto = texto_completo_de_palavras(palavras)
    sentencas_texto = sent_tokenize(texto, language=idioma)

    sentencas: List[Sentenca] = []
    idx_palavra = 0           

    for i, sent in enumerate(sentencas_texto):
        tokens_sent = sent.split()
        if not tokens_sent:
            continue

        inicio_sent = None
        fim_sent = None
        probs = []
        palavras_usadas = 0

        # Avança o ponteiro de palavras até cobrir todos os tokens da sentença
        while palavras_usadas < len(tokens_sent) and idx_palavra < len(palavras):
            pw = palavras[idx_palavra]
            # Remove pontuação para comparar
            token_limpo = re.sub(r"[^\w]", "", tokens_sent[palavras_usadas]).lower()
            palavra_limpa = re.sub(r"[^\w]", "", pw["palavra"]).lower()

            if token_limpo == palavra_limpa or token_limpo in palavra_limpa or palavra_limpa in token_limpo:
                if inicio_sent is None:
                    inicio_sent = pw["inicio"]
                fim_sent = pw["fim"]
                probs.append(pw["probabilidade"])
                palavras_usadas += 1

            idx_palavra += 1

        if inicio_sent is None:
            continue  # sentença sem palavras mapeadas — pula

        confianca = sum(probs) / len(probs) if probs else 0.0

        sentencas.append(Sentenca(
            id=i + 1,
            texto=sent.strip(),
            inicio=round(inicio_sent, 3),
            fim=round(fim_sent, 3),
            inicio_fmt=segundos_para_hhmss(inicio_sent),
            fim_fmt=segundos_para_hhmss(fim_sent),
            duracao=round(fim_sent - inicio_sent, 3),
            num_palavras=len(tokens_sent),
            num_caracteres=len(sent.strip()),
            confianca_media=round(confianca, 4),
        ))

    # Divide sentenças longas
    resultado_final = []
    for s in sentencas:
        resultado_final.extend(dividir_sentenca_longa(s, palavras, max_duracao=30.0))
    
    # Renumera os ids
    for i, s in enumerate(resultado_final):
        s.id = i + 1

    return resultado_final

#Saídas

def salvar_json(sentencas: List[Sentenca], caminho: str):
    dados = [asdict(s) for s in sentencas]
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
    print(f"JSON salvo em: {caminho}")

def salvar_txt(sentencas: List[Sentenca], caminho: str):
    linhas = []
    for s in sentencas:
        linhas.append(
            f"[{s.id:03d}] {s.inicio_fmt} → {s.fim_fmt}  "
            f"({s.duracao:.1f}s | {s.num_palavras} palavras | confiança {s.confianca_media:.0%})\n"
            f"      {s.texto}\n"
        )
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))
    print(f"TXT salvo em: {caminho}")

def salvar_srt(sentencas: List[Sentenca], caminho: str):
    blocos = []
    for s in sentencas:
        blocos.append(
            f"{s.id}\n"
            f"{s.inicio_fmt.replace(',', ',')} --> {s.fim_fmt.replace(',', ',')}\n"
            f"{s.texto}\n"
        )
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("\n".join(blocos))
    print(f"SRT salvo em: {caminho}")

def imprimir_resumo(sentencas: List[Sentenca]):
    print("\n" + "═" * 70)
    print(f"RESUMO  —  {len(sentencas)} sentenças encontradas")
    print("═" * 70)
    for s in sentencas[:10]:   # mostra as 10 primeiras no terminal
        print(f"  [{s.id:03d}] {s.inicio_fmt} → {s.fim_fmt}  ({s.duracao:.1f}s)")
        print(f"        {s.texto[:80]}{'…' if len(s.texto) > 80 else ''}")
    if len(sentencas) > 10:
        print(f"  … e mais {len(sentencas) - 10} sentenças (veja os arquivos de saída)")
    print("═" * 70)

#baixar video
def baixar_video(url: str, saida: str = "video.mp4") -> str:
    import yt_dlp
    print(f"Baixando vídeo de: {url}")
    opcoes = {
        "outtmpl": saida,
        "format": "mp4",
    }
    with yt_dlp.YoutubeDL(opcoes) as ydl:
        ydl.download([url])
    print(f"Vídeo salvo em: {saida}")
    return saida



#Input
def processar_video(
    caminho_video: str = None,
    url: str = None,          # ← novo
    modelo_whisper: str = "base",
    idioma: str = "pt",
    idioma_nltk: str = "portuguese",
    manter_audio: bool = False,
):
    if url:
        caminho_video = baixar_video(url)

    video_path = Path(caminho_video)
    if not video_path.exists(): 
        sys.exit(f"Arquivo nao encontrado: {caminho_video}")

    base = video_path.stem
    audio_temp = f"{base}_audio_temp.wav"

    #1. Extrair _audio
    extrair_audio(caminho_video, audio_temp)

    # 2. Transcrever
    resultado = transcrever_audio(audio_temp, modelo=modelo_whisper, idioma=idioma)

    # 3. Dividir em sentenças
    palavras = segmentos_para_palavras(resultado)
    sentencas = dividir_em_sentencas(palavras, idioma=idioma_nltk)

    # 4. Salvar resultados
    salvar_json(sentencas, f"{base}_sentencas.json")
    salvar_txt(sentencas, f"{base}_sentencas.txt")
    salvar_srt(sentencas, f"{base}_sentencas.srt")

    #Resumo no terminal
    imprimir_resumo(sentencas)

    #Limpar audio temporario
    if not manter_audio: 
        Path(audio_temp).unlink(missing_ok=True)

    return sentencas

#CLI Simples

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extrai áudio de vídeo, transcreve e divide em sentenças com timestamps."
    )

    parser.add_argument("video", nargs="?", default=None, help="Caminho para o arquivo de vídeo")

    parser.add_argument(
        "--modelo", default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Modelo Whisper (padrão: base)"
    )

    parser.add_argument(
        "--idioma", default="pt",
        help="Código ISO do idioma (padrão: pt). Use 'auto' para detecção automática."
    )

    parser.add_argument(
        "--manter-audio", action="store_true",
        help="Mantém o arquivo WAV temporário após a transcrição"
    )

    parser.add_argument(
        "--url",
        help="URL do YouTube para baixar e transcrever"
    )

    args = parser.parse_args()
    idioma = None if args.idioma == "auto" else args.idioma
    
    #Mapeamento simplificado codigo -> Nome NTLK
    mapa_nltk = {
        "pt": "portuguese", "en": "english", "es": "spanish",
        "fr": "french", "de": "german", "it": "italian",
        "nl": "dutch", "pl": "polish", "ru": "russian",
    }

    idioma_nltk = mapa_nltk.get(idioma, "portuguese")

    processar_video(
        caminho_video=args.video,
        url=args.url,
        modelo_whisper=args.modelo,
        idioma=idioma,
        idioma_nltk=idioma_nltk,
        manter_audio=args.manter_audio,
    )