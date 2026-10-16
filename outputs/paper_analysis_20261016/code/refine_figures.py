"""Re-render S1/S5 from saved plot data; no reread of 120 prediction arrays."""
from pathlib import Path
p=Path('/ceph_data/yeonwoo/MHW_subset/outputs/paper_analysis_20261016/code/analyze.py')
s=p.read_text()
s=s.replace("gg=diag.groupby('epoch')[k]", "gg=diag[diag.epoch.isin([1,5,10,20,80])].groupby('epoch')[k]")
s=s.replace('Saved auxiliary diagnostic batches: median and interquartile range across available runs at each diagnostic epoch. Sampling is irregular between scheduled epochs; epoch-specific counts are provided.', 'Saved auxiliary diagnostic batches: median and interquartile range at common scheduled epochs 1, 5, 10, 20 and 80, each with all 24 runs. Irregular best-candidate diagnostics are retained in CSV but excluded from the plotted aggregate trend; epoch-specific counts are provided.')
s=s.replace("for ax in axs[i]:ax.set(xlabel='Longitude (°E)',ylabel='Latitude (°N)')", "for j,ax in enumerate(axs[i]):ax.set(xlabel='Longitude (°E)',ylabel='Latitude (°N)' if j==0 else '')")
p.write_text(s)
exec(compile(s[:s.index("log('Read existing summaries")],str(p),'exec'))
diag=readcsv(V/'aux_diagnostics_rechecked.csv')
exec(compile(s[s.index('# S1 fixed'):s.index('# S2 all')],str(p),'exec'))
counts=diag[diag.epoch.isin([1,5,10,20,80])].groupby('epoch').name.nunique();assert (counts==24).all()
case=pd.read_csv(O/'plot_data/case_all_models.csv');lat=np.sort(case.latitude.unique());lon=np.sort(case.longitude.unique());example={}
for (lead,m),g in case.groupby(['lead','model']):
 def grid(k):return g.pivot(index='latitude',columns='longitude',values=k).reindex(index=lat,columns=lon).to_numpy()
 obs=grid('obs_sst');example[(lead,m)]={'obs':obs,'pred':grid('pred_sst'),'pm':grid('pred_mhw')==1,'om':grid('obs_mhw')==1,'date':g.date.iloc[0]}
sea0=np.isfinite(example[(7,'Baseline')]['obs']);assert sea0.sum()==661
exec(compile(s[s.index('# S5 neutral'):s.index('allcases=[]')],str(p),'exec'))
m=pd.read_csv(O/'figure_table_manifest.csv')
for row in manifest:
 ix=m.index[m.id==row['id']];assert len(ix)==1
 m.loc[ix,'caption']=row['caption']
m.to_csv(O/'figure_table_manifest.csv',index=False)
inv=J(O/'reports/source_inventory.json');source(p);source(Path(__file__));inv.update(sources);savejson('reports/source_inventory.json',inv)
vv=J(O/'reports/output_validation.json');vv['S1_common_epoch_run_counts']={str(k):int(v) for k,v in counts.items()};vv['visually_reviewed']=['Figure1','Figure3','Figure5','S1','S5'];savejson('reports/output_validation.json',vv)
from PIL import Image
for stem in ['S1_auxiliary_decay','S5_calendar_case']:
 for ext in ['png','pdf','svg']:assert (O/f'figures/supplementary/{stem}.{ext}').stat().st_size>100
 with Image.open(O/f'figures/supplementary/{stem}.png') as im:assert min(im.info['dpi'])>=300
write('captions/all_captions.md','\n\n'.join((O/'captions'/(row.id+'.md')).read_text() for row in m.itertuples() if row.kind=='figure'))
write('README.md','''# Oracle120 final paper analysis\n\nStart with [Korean analysis summary](reports/analysis_summary_ko.md), [claim–evidence matrix](reports/claim_evidence_matrix.md), [limitations](reports/limitations.md), and [reproducibility](reports/reproducibility_summary.md).\n\n- [Artifact manifest](figure_table_manifest.csv): 5 main and 6 supplementary figures; Table1–6 plus separate Validation table.\n- [English captions](captions/all_captions.md); figures in PNG (350dpi), PDF and SVG.\n- `tables/`: CSV and LaTeX; detailed Test/Validation seed metrics and regional summaries.\n- `statistics/`: six paired comparisons; original 90-day bootstrap; physical/seasonal summaries.\n- `plot_data/`: grid errors, physical histograms/maps, regional scores, neutral calendar case.\n- `reports/output_validation.json`: numerical and format checks.\n\nLaTeX tables use booktabs/longtable; UTF-8 symbols are supported with XeLaTeX/LuaLaTeX. Tables with many columns may be placed in a landscape supplement or reformatted by the target journal. The CSV files retain full precision.\n\nFull analysis: `code/analyze.py`; final S1/S5 display refinement can be repeated with `code/refine_figures.py`. Both write only within this new analysis directory, except the required append to the shared research log. No training, inference, new bootstrap or prior audit rerun.\n''')
now=datetime.datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
with (R/'docs/research_log.md').open('a') as f:f.write(f'\n- {now} — paper_analysis_20261016 그림 최종 점검: S1은 24개 실행이 공통으로 존재하는 epoch1/5/10/20/80만 중앙값/IQR 추세로 표시하고 불규칙 기록은 CSV에 보존. S5는 반복 위도 축 라벨을 줄여 colorbar 겹침 개선. 저장 plot CSV에서 재렌더링; 원본 예측·평가 변경 없음. 통합 캡션/README 및 manifest·코드해시 갱신.\n')
print('REFINEMENT COMPLETE',vv,flush=True)
