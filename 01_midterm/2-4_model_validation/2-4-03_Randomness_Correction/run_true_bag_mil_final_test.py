"""One final held-test comparison for the already-selected MIL configuration."""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sentence_transformers import SentenceTransformer

HERE=Path(__file__).resolve().parent
SRC=HERE/'run_true_bag_mil_fusion.py'
spec=importlib.util.spec_from_file_location('mil_source',SRC); m=importlib.util.module_from_spec(spec); sys.modules['mil_source']=m; spec.loader.exec_module(m)
OUT=HERE/'true_bag_mil'; TARGET=m.TARGET

def metrics(name,y,s):
 n=max(1,len(y)//10); top=np.argsort(-s)[:n]
 return {'split':'test','model':name,'pr_auc':float(average_precision_score(y,s)),
         'roc_auc':float(roc_auc_score(y,s)),'recall_at_10pct':float(y[top].sum()/y.sum())}

panel=pd.read_parquet(m.PANEL_PATH).reset_index(drop=True)
panel.parent_asin=panel.parent_asin.astype(str); panel.year_month=panel.year_month.astype(str)
panel['row_id']=np.arange(len(panel)); panel=m.base.assign_random_group_split(panel)
future={'is_review_volume_drop','is_review_volume_drop_2m','next_review_count','expected_next_review_count','next_review_volume_ratio','next_2m_review_count','expected_next_2m_review_count'}
bad={'parent_asin','year_month','year','random_stratum','random_fold','random_split','row_id'}|future
feats=[c for c in panel if c not in bad and pd.api.types.is_numeric_dtype(panel[c])]

revs=pd.read_parquet(m.ROOT/'data/interim/clean_reviews.parquet',columns=['parent_asin','year_month','review_datetime_utc','text_norm','rating'])
revs.parent_asin=revs.parent_asin.astype(str); revs.year_month=revs.year_month.astype(str); revs.text_norm=revs.text_norm.fillna('').astype(str)
revs=revs[revs.text_norm.str.len()>=3].merge(panel[['row_id','parent_asin','year_month',TARGET]],on=['parent_asin','year_month'],how='inner')
panel=m.base.build_month_documents(panel,revs)
train_valid=panel[panel.random_split.ne('test')].reset_index(drop=True)
test=panel[panel.random_split.eq('test')].reset_index(drop=True)

# Meta models are trained only on the saved Train OOF scores, never on Valid/Test.
oof=pd.read_parquet(OUT/'train_oof_expert_scores.parquet')
def meta(cols):
 return LogisticRegression(class_weight='balanced',C=.2,max_iter=1000,random_state=m.SEED).fit(oof[cols],oof.target)
raw_meta=meta(['s45','raw']); full_meta=meta(['s45','raw','mil'])

print('Test experts: Train+Valid으로 재학습 (Test 라벨 미사용)')
pred,_=m.base.predict_base_experts(train_valid,test,revs,feats)
encoder=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
xtr,mtr=m.embed_bags(train_valid,revs,encoder); xte,mte=m.embed_bags(test,revs,encoder)
mil=m.fit_predict(xtr,mtr,train_valid[TARGET].to_numpy(),xte,mte)
x=pd.DataFrame({'s45':pred.s45_score,'raw':pred.context_score,'mil':mil})
s45raw=raw_meta.predict_proba(x[['s45','raw']])[:,1]
full=full_meta.predict_proba(x[['s45','raw','mil']])[:,1]
y=test[TARGET].to_numpy()
report=pd.DataFrame([metrics('S45',y,pred.s45_score),metrics('raw_full_context',y,pred.context_score),metrics('bag_mil',y,mil),metrics('S45_raw_stack',y,s45raw),metrics('S45_raw_MIL_stack',y,full)])
base=float(report.loc[report.model.eq('S45_raw_stack'),'pr_auc'].iloc[0]); final=float(report.loc[report.model.eq('S45_raw_MIL_stack'),'pr_auc'].iloc[0])
report['pr_auc_delta_vs_S45_raw_stack']=report.pr_auc-base
report.to_csv(OUT/'final_test_mil_ablation_metrics.csv',index=False)
pd.DataFrame({'row_id':test.row_id,'parent_asin':test.parent_asin,'year_month':test.year_month,TARGET:y,'s45_raw_score':s45raw,'s45_raw_mil_score':full,'mil_increment':full-s45raw}).to_parquet(OUT/'final_test_mil_predictions.parquet',index=False)
summary={'experiment':'final_test_MIL_ablation','test_data_used':True,'test_note':'Test was previously inspected for raw-context evaluation; report as MIL augmented comparison, not pristine first-use test.','test_S45_raw_stack_pr_auc':base,'test_S45_raw_MIL_stack_pr_auc':final,'MIL_increment_over_S45_raw':final-base,'valid_increment_preselected':0.003539760326945307}
(OUT/'final_test_mil_ablation_summary.json').write_text(__import__('json').dumps(summary,indent=2))
print(__import__('json').dumps(summary,indent=2))
