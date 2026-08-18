"""True bag-level MIL: Test is deliberately never loaded into modelling."""
from __future__ import annotations
import gc, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
import torch
from torch import nn
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
import run_random_group_context_fusion as base
ROOT=next((q for q in HERE.parents if (q/'data/interim/clean_reviews.parquet').exists()), None)
if ROOT is None:
    raise FileNotFoundError(f'프로젝트 루트를 찾지 못했습니다: {HERE}')
PANEL_PATH=next((q for q in [
    ROOT/'data/processed/product_month_review_volume_2m_labeled.parquet',
    ROOT/'data/processed/product_month_review_volume_labeled.parquet',
] if q.exists()), None)
if PANEL_PATH is None:
    raise FileNotFoundError(f'라벨 패널 파일이 없습니다: {ROOT}/data/processed')
OUT=HERE/'true_bag_mil'; OUT.mkdir(exist_ok=True)
TARGET=base.TARGET; SEED=42; MAX_REVIEWS=8; EPOCHS=int(os.getenv('MIL_EPOCHS','18'))
torch.manual_seed(SEED); np.random.seed(SEED)

class MIL(nn.Module):
    def __init__(self,d):
        super().__init__(); self.h=nn.Sequential(nn.Linear(d+3,128),nn.ReLU(),nn.Dropout(.20),nn.Linear(128,64),nn.ReLU())
        self.a=nn.Sequential(nn.Linear(64,32),nn.Tanh(),nn.Linear(32,1)); self.o=nn.Linear(64*3,1)
    def forward(self,x,mask):
        h=self.h(x); a=torch.sigmoid(self.a(h)).squeeze(-1)*mask
        den=a.sum(1,keepdim=True).clamp_min(1e-6); att=(h*a.unsqueeze(-1)).sum(1)/den
        mean=(h*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        top=(h.masked_fill(~mask.bool().unsqueeze(-1),-1e4)).topk(min(3,h.shape[1]),1).values.mean(1)
        return self.o(torch.cat([mean,top,att],1)).squeeze(-1),a

def embed_bags(rows,revs,encoder):
    r=revs[revs.row_id.isin(rows.row_id)].sort_values(['row_id','review_datetime_utc']).groupby('row_id',sort=False).tail(MAX_REVIEWS)
    texts=r.text_norm.tolist(); emb=encoder.encode(texts,batch_size=96,show_progress_bar=True,normalize_embeddings=True).astype('float32') if texts else np.zeros((0,384),'float32')
    r=r[['row_id','rating','text_norm']].copy(); r['i']=r.groupby('row_id').cumcount(); r['emb']=list(emb)
    d=emb.shape[1] if len(emb) else 384; X=np.zeros((len(rows),MAX_REVIEWS,d+3),'float32'); M=np.zeros((len(rows),MAX_REVIEWS),'float32'); mp={v:i for i,v in enumerate(rows.row_id)}
    for z in r.itertuples():
        j=mp[z.row_id]; e=z.emb; X[j,z.i,:d]=e; X[j,z.i,d:]=[z.rating/5,len(z.text_norm)/500,z.rating<=2]; M[j,z.i]=1
    return X,M

def fit_predict(xtr,mtr,y,xev,mev):
    dev='mps' if torch.backends.mps.is_available() else 'cpu'; model=MIL(xtr.shape[-1]-3).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4)
    pos=(len(y)-y.sum())/max(y.sum(),1); crit=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos,dtype=torch.float32,device=dev))
    xt=torch.tensor(xtr,device=dev); mt=torch.tensor(mtr,device=dev); yt=torch.tensor(y.astype('float32'),device=dev)
    model.train()
    for _ in range(EPOCHS): opt.zero_grad(); loss=crit(model(xt,mt)[0],yt); loss.backward(); opt.step()
    model.eval();
    with torch.no_grad(): return torch.sigmoid(model(torch.tensor(xev,device=dev),torch.tensor(mev,device=dev))[0]).cpu().numpy()

def pr(y,s): return float(average_precision_score(y,s))
def main():
 panel=pd.read_parquet(PANEL_PATH).reset_index(drop=True); panel['parent_asin']=panel.parent_asin.astype(str); panel['year_month']=panel.year_month.astype(str); panel['row_id']=np.arange(len(panel)); panel=base.assign_random_group_split(panel); panel=panel[panel.random_split.ne('test')].copy() # hard exclusion
 future={'is_review_volume_drop','is_review_volume_drop_2m','next_review_count','expected_next_review_count','next_review_volume_ratio','next_2m_review_count','expected_next_2m_review_count'}; bad={'parent_asin','year_month','year','random_stratum','random_fold','random_split','row_id'}|future
 feats=[c for c in panel if c not in bad and pd.api.types.is_numeric_dtype(panel[c])]
 revs=pd.read_parquet(ROOT/'data/interim/clean_reviews.parquet',columns=['parent_asin','year_month','review_datetime_utc','text_norm','rating']); revs.parent_asin=revs.parent_asin.astype(str); revs.year_month=revs.year_month.astype(str); revs.text_norm=revs.text_norm.fillna('').astype(str); revs=revs[revs.text_norm.str.len()>=3].merge(panel[['row_id','parent_asin','year_month',TARGET]],on=['parent_asin','year_month'],how='inner')
 panel=base.build_month_documents(panel, revs)
 train=panel[panel.random_split.eq('train')].reset_index(drop=True); valid=panel[panel.random_split.eq('valid')].reset_index(drop=True); enc=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu')
 xt,mt=embed_bags(train,revs,enc); xv,mv=embed_bags(valid,revs,enc); oof=np.zeros(len(train)); sg=StratifiedGroupKFold(5,shuffle=True,random_state=43)
 for k,(a,b) in enumerate(sg.split(train,y=train[TARGET],groups=train.parent_asin),1): print(f'OOF {k}/5'); oof[b]=fit_predict(xt[a],mt[a],train.iloc[a][TARGET].to_numpy(),xt[b],mt[b]); gc.collect()
 s45o=np.zeros(len(train)); ctxo=np.zeros(len(train))
 for a,b in sg.split(train,y=train[TARGET],groups=train.parent_asin):
  pp,_=base.predict_base_experts(train.iloc[a],train.iloc[b],revs,feats); s45o[b]=pp.s45_score; ctxo[b]=pp.context_score
 pp,_=base.predict_base_experts(train,valid,revs,feats)
 milv=fit_predict(xt,mt,train[TARGET].to_numpy(),xv,mv)

 oof_experts=pd.DataFrame({
     'row_id':train.row_id, 'target':train[TARGET],
     's45':s45o, 'raw':ctxo, 'mil':oof,
 })
 valid_experts=pd.DataFrame({
     'row_id':valid.row_id, 'target':valid[TARGET],
     's45':pp.s45_score, 'raw':pp.context_score, 'mil':milv,
 })
 oof_experts.to_parquet(OUT/'train_oof_expert_scores.parquet',index=False)
 valid_experts.to_parquet(OUT/'valid_expert_scores.parquet',index=False)

 def stack_eval(name, cols):
     clf=LogisticRegression(
         class_weight='balanced', C=.2, max_iter=1000, random_state=SEED
     ).fit(oof_experts[cols],oof_experts.target)
     score=clf.predict_proba(valid_experts[cols])[:,1]
     return name,score,dict(zip(cols,clf.coef_[0])),float(clf.intercept_[0])

 stack_out=[
     stack_eval('S45_raw_stack',['s45','raw']),
     stack_eval('S45_MIL_stack',['s45','mil']),
     stack_eval('raw_MIL_stack',['raw','mil']),
     stack_eval('S45_raw_MIL_stack',['s45','raw','mil']),
 ]
 final=next(score for name,score,_,_ in stack_out if name=='S45_raw_MIL_stack')
 def row(n,s):
  y=valid[TARGET].to_numpy(); top=np.argsort(-s)[:max(1,len(s)//10)]; return {'model':n,'pr_auc':pr(y,s),'roc_auc':float(roc_auc_score(y,s)),'recall_at_10pct':float(y[top].sum()/y.sum())}
 out=pd.DataFrame(
     [row('S45',pp.s45_score),row('raw_full_context',pp.context_score),row('bag_mil',milv)]
     +[row(name,score) for name,score,_,_ in stack_out]
 )
 base_pr=float(out.loc[out.model.eq('S45_raw_stack'),'pr_auc'].iloc[0])
 full_pr=float(out.loc[out.model.eq('S45_raw_MIL_stack'),'pr_auc'].iloc[0])
 out['pr_auc_delta_vs_S45_raw_stack']=out.pr_auc-base_pr
 out.to_csv(OUT/'true_bag_mil_metrics.csv',index=False)

 pd.DataFrame([
     {'model':name,'features':','.join(coefs),'intercept':intercept,
      **{f'coef_{key}':value for key,value in coefs.items()}}
     for name,_,coefs,intercept in stack_out
 ]).to_csv(OUT/'stacking_coefficients.csv',index=False)

 s45raw=next(score for name,score,_,_ in stack_out if name=='S45_raw_stack')
 audit=valid[['parent_asin','year_month',TARGET]].copy()
 audit['s45_raw_score']=s45raw
 audit['s45_raw_mil_score']=final
 audit['mil_score']=milv
 audit['mil_increment']=final-s45raw
 audit.sort_values('mil_increment',ascending=False).to_csv(
     OUT/'valid_mil_increment_audit.csv',index=False
 )

 summary={
     'experiment':'true_bag_sigmoid_attention_mil_ablation',
     'test_data_used':False,
     'valid_S45_raw_stack_pr_auc':base_pr,
     'valid_S45_raw_MIL_stack_pr_auc':full_pr,
     'MIL_increment_over_fair_baseline':full_pr-base_pr,
     'selection_threshold_pr_auc':0.003,
     'decision':'adopt_MIL' if full_pr-base_pr>=.003 else 'keep_S45_raw',
 }
 (OUT/'true_bag_mil_summary.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
