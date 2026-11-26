"""
VinDR_prepare_neg.py - anotacoes do VinDr com imagens SEM achado no treino

Usa o VinDR_prepare.py sem altera-lo: so troca a proporcao de negativas e a
pasta de saida. A divisao das imagens com massa (treino/val/teste, semente 42)
e a mesma; as negativas ("No Finding", sem nenhum achado) entram so no treino,
de estudos do split "training" fora da validacao.

Saida: annotations_VinDR_neg1/train.json (+ val.json e test.json, que devem
ser identicos aos de annotations_VinDR; o script confere).

Uso:  python VinDR_prepare_neg.py
"""
import filecmp
import os

import VinDR_prepare as prep

prep.NEGATIVOS_POR_POSITIVO = 1.0          # 1 negativa por imagem com massa
prep.SAIDA = 'annotations_VinDR_neg1'

if __name__ == '__main__':
    prep.main()
    for nome in ('val.json', 'test.json'):
        igual = filecmp.cmp(os.path.join(prep.SAIDA, nome),
                            os.path.join('annotations_VinDR', nome), shallow=False)
        print(f'{nome}: {"IGUAL" if igual else "DIFERENTE"} ao de annotations_VinDR')
        if not igual:
            raise SystemExit('ERRO: a divisao mudou; nao use estas anotacoes.')
