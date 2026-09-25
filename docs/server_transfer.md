# 추가 실험 서버 전송

- 접속: `ssh -p 20001 yeonwoo@203.253.207.61`
- 서버 작업 경로: `/ceph_data/yeonwoo/MHW_subset`
- 전송 파일: 로컬 `transfer/MHW_subset_server.tar`
- 포함: src, scripts, data의 NetCDF와 v2 subset 3개, docs, README.md, AGENTS.md, requirements.txt, .gitattributes, .gitignore, 기존 Full의 JSON/CSV/Markdown.
- 제외: Git 이력, 가상환경, 캐시, 기존 Full 체크포인트·예측 배열·실행 로그. Full 모델 재평가가 필요하면 해당 파일을 별도로 복사한다.
- 원본을 이동하거나 삭제하지 않고 복사본을 묶는다. 파일별 SHA-256은 압축 파일 안의 `TRANSFER_SHA256SUMS`에 기록한다.

## 전송

서버 터미널에서:

```bash
df -h /ceph_data/yeonwoo
mkdir -p /ceph_data/yeonwoo/MHW_subset
```

내 PC의 새 PowerShell에서:

```powershell
scp -P 20001 "C:\Users\hyun\Desktop\MHW_subset\transfer\MHW_subset_server.tar" yeonwoo@203.253.207.61:/ceph_data/yeonwoo/
```

서버 터미널에서 아래 명령은 새로 만든 빈 프로젝트 폴더를 대상으로 실행한다. 같은 이름의 기존 파일이 있으면 먼저 확인한다.

```bash
tar -xf /ceph_data/yeonwoo/MHW_subset_server.tar -C /ceph_data/yeonwoo/MHW_subset
cd /ceph_data/yeonwoo/MHW_subset
sha256sum -c TRANSFER_SHA256SUMS
```

압축 파일과 해제본이 함께 있으므로 약 2.7 GB 이상의 여유 공간이 필요하며, 학습 출력 공간은 별도로 확보한다. Ceph의 개인 quota는 관리자에게 확인한다.

## 전송 이후

- 서버 Python/GPU 환경과 requirements.txt 의존성을 확인한다. requirements.txt는 기존 Full 환경 lockfile이 아니다.
- 환경 준비 후 `python -B scripts/make_train_subsets.py --check`로 subset 재현성을 확인한다.
- v2 subset의 학습 연결은 아직 구현되지 않았다. 현재 학습 진입점을 그대로 실행하면 기존 전체 학습 분할을 사용한다.
- 학습량 기준과 정규화 통계 적용 방식을 먼저 결정해야 한다. 모델 코드와 기존 분할은 유지한다.
- 기존 실행기는 PowerShell 파일이며 Linux 실행 방법은 별도 준비가 필요하다.
- 원본 코드와 Full JSON에 남은 `D:/MHW` 경로는 보존한다. 실행 시 현재 데이터·출력 경로를 명시하고, 새 결과는 `outputs/data_efficiency/` 등 별도 폴더에 기록한다.

이 문서는 전송 준비 안내이며 실제 서버 전송이나 학습 완료를 의미하지 않는다.
