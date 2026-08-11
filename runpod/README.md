# Entrenamiento en RunPod — `codice_geo`

Infraestructura para entrenar el LoRA del paso 5 del brief con `kohya_ss`.

## Por qué kohya y no los notebooks

Los notebooks de `LoRA_train_workflows/01_SDXL/` usan
`train_text_to_image_lora_sdxl.py` de diffusers, que emite el LoRA en formato
PEFT y necesita conversión antes de que ComfyUI lo cargue — de ahí que exista
`04_SDXL_LoRA_Convert_Use.ipynb`. kohya emite directamente el `.safetensors`
que ComfyUI abre, y además soporta *repeats* por carpeta, que es como se
equilibran las cuatro clases del dataset. El brief ya especificaba kohya.

## Elección de GPU

| GPU | VRAM | ~USD/h | Sirve |
|---|---:|---:|---|
| **RTX 4090** | 24 GB | 0.34–0.44 | Sí — la opción por defecto |
| RTX A6000 | 48 GB | 0.76 | Sí, sin ventaja real aquí |
| A100 80 GB | 80 GB | 1.19 | Sólo si quieres batch > 2 |

Con `gradient_checkpointing` y `AdamW8bit`, SDXL a 1024 con network dim 32 cabe
holgado en 24 GB. Una corrida de ~350 imágenes son unos 3.500 pasos por época:
**40–60 minutos en una 4090**.

Sobre el presupuesto: el brief asignó $6,000 MXN (~$320 USD) para ~200 h en dos
iteraciones. Dos iteraciones con experimentación generosa difícilmente pasan de
20 h, o sea **~$8 USD**. El renglón está sobredimensionado y conviene decírselo
a Carlos por si quiere reasignarlo a las tandas de generación en ComfyUI.

## Network Volume

Crea un **Network Volume de 100 GB** y móntalo en `/workspace`. Es lo que hace
que esto salga barato: el checkpoint de SDXL (~7 GB), el dataset y las salidas
persisten al apagar el pod. Apagas y dejas de pagar cómputo; el volumen cuesta
~$7/mes aparte.

Sin volumen, cada pod nuevo vuelve a descargar 7 GB de modelo.

## Puesta en marcha

```bash
# 1. Pod nuevo con el template runpod/pytorch:2.4.0-py3.11-cuda12.4.1
#    y el Network Volume montado en /workspace

git clone <este-repo> /workspace/codice && cd /workspace/codice/runpod

# 2. Instala kohya y baja los modelos base (unos minutos la primera vez)
bash bootstrap.sh

# 3. Conecta el Drive del proyecto (sólo la primera vez por volumen)
bash sync.sh setup     # elige "Use auto config: n" — el pod no tiene navegador
bash sync.sh pull      # trae 02_DATASET/train

# 4. Valida el árbol antes de gastar una hora de GPU
bash train.sh --smoke  # 10 pasos

# 5. Corrida real
bash train.sh

# 6. Sube los pesos a 03_LORA
bash sync.sh push
```

El paso 4 no es opcional. Si a alguna imagen le falta su `.txt`, kohya entrena
con caption vacío **sin avisar**, y el fallo sólo se nota al generar. `train.sh`
compara el número de imágenes contra el de captions en cada carpeta y avisa.

## Parámetros y por qué

Todo en `train_codice_geo.toml`. Los que no son obvios:

- **`enable_bucket`** — agrupa por relación de aspecto en vez de recortar al
  centro. Crítico aquí: recortar al centro un afloramiento estratificado se come
  justo la estratificación, que es lo que se quiere aprender.
- **`flip_aug = false`** — el volteo horizontal duplicaría el dataset, pero la
  estratigrafía tiene orientación real. Entrenar con capas invertidas enseña una
  geología imposible.
- **`keep_tokens = 1`** — `shuffle_caption` baraja los tags para que el modelo no
  memorice su orden, pero el trigger `codice_geo` debe quedarse fijo al frente.
- **`min_snr_gamma = 5.0`** — pondera la pérdida por relación señal-ruido.
  Acelera la convergencia en texturas de alta frecuencia como la roca. Es el
  mismo valor que ya usa `03_SDXL_LoRA_Train.ipynb`.
- **`network_train_unet_only = false`** — sin entrenar el text encoder el trigger
  queda débil y hay que subir mucho el peso del LoRA al generar.
- **VAE fp16-fix** — el VAE original de SDXL produce NaN en fp16. Los notebooks
  del repo ya usan este mismo fork.

## Después del entrenamiento

Los `.safetensors` van a `03_LORA/` del Drive. Para ComfyUI (paso 8 del brief):
SDXL base → LoRA `codice_geo` a 0.8–1.0 → ControlNet depth 0.7–0.8 + ControlNet
normal 0.5–0.6 en paralelo → DPM++ 2M Karras, 30 pasos, CFG 7.

Los mapas de depth y normal salen de los `.blend` que ya están en
`01_ESCANEO/SIMULACIONES` (Z-pass y Normal-pass, pasos 6–7 del brief).
