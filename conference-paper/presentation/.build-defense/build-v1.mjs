import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile, FileBlob } from '@oai/artifact-tool';

const workspaceDir = 'C:/work/学校/A.SPBU/paper/ZhaoCHI/Paper/Master/conference-paper/presentation';
const tmp = path.join(workspaceDir, '.build-defense');
const SKILL_DIR = 'C:/Users/swj17/.codex/plugins/cache/openai-primary-runtime/presentations/26.915.20218/skills/presentations';
const RUNTIME_PYTHON = 'C:/Users/swj17/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES='C:/Users/swj17/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const { finalizePresentation } = await import(pathToFileURL(path.join(SKILL_DIR,'container_tools/artifact_tool_utils.mjs')).href);
const C = { bg:'#F6F3EB', navy:'#0B2033', teal:'#0C7C78', mint:'#75D0C9', red:'#C63E40', gray:'#526675', light:'#D7DEDF', white:'#FFFFFF', amber:'#B77818' };
const scale=4/3;
const presentation=Presentation.create({slideSize:{width:1600,height:900}});
let count=0;
const nativeTableSlides=[];
const notes=[];
function text(s, value, x, y, w, h, size=25, color=C.navy, bold=false, align='left') {
  const a=s.shapes.add({geometry:'textbox',name:`text-${++count}`,position:{left:x*scale,top:y*scale,width:w*scale,height:h*scale},fill:'none',line:{fill:'none',width:0}});
  a.text=value;
  a.text.style={typeface:'Arial',fontSize:size*scale,color,bold,alignment:align,autoFit:'none',verticalAlignment:'middle',wrap:true};
  return a;
}
function line(s,x,y,w,color=C.teal,width=1.5){s.shapes.add({geometry:'line',name:`rule-${++count}`,position:{left:x*scale,top:y*scale,width:w*scale,height:0},fill:'none',line:{fill:color,width:width*scale}});}
function slide(title,{dark=false,tag=''}={}){
  const s=presentation.slides.add(); s.background.fill=dark?C.navy:C.bg;
  const i=presentation.slides.items.length;
  if(title){text(s,title,60,44,1090,72,39,dark?C.white:C.navy,true);line(s,60,134,1080,dark?C.mint:C.teal);}
  text(s,String(i).padStart(2,'0'),1112,636,36,24,16,dark?C.mint:C.gray,false,'right');
  if(tag) text(s,tag,60,628,1000,29,16,dark?C.mint:C.gray);
  return s;
}
function note(s,cn,sources){
 const timings=[30,55,45,55,60,60,60,85,50,65,65,70,55,50,45,35];
 if(notes.length<16) cn=cn.replace(/建议用时 \d+ 秒/,`建议用时 ${timings[notes.length]} 秒`);
 const root='C:/work/学校/A.SPBU/paper/ZhaoCHI/Paper/';
 sources=sources.map(v=>v.replace('../../../../Master-mr2/',root+'Master-mr2/').replace('../../configs/',root+'Master/configs/').replace('../../README.md',root+'Master/README.md').replace('../paper/',root+'Master/conference-paper/paper/'));
 const n=`讲述提示\n${cn}\n\n资料来源\n${sources.join('\n')}`;s.speakerNotes.textFrame.setText(n);notes.push(n);
}
function pair(s,x,y,w,head,body,{color=C.teal,size=25}={}){text(s,head,x,y,w,40,28,color,true);text(s,body,x,y+55,w,135,size);}
function row(s,n,y,head,body){text(s,n,60,y,70,45,28,C.teal,true);text(s,head,145,y,985,43,28,C.navy,true);text(s,body,145,y+49,985,71,24,C.gray);}
function table(s,values,widths,{x=60,y=190,h=340,font=21}={}){
 const w=widths.reduce((a,b)=>a+b,0);
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x*scale,top:y*scale,width:w*scale,height:h*scale,columnWidths:widths.map(v=>v*scale),values});
 t.borders.assign({fill:C.light,width:0.7*scale,style:'solid'});
 for(let r=0;r<values.length;r++){
   t.rows[r].height=h/values.length*scale;
   for(let c=0;c<values[0].length;c++){
     const a=t.getCell(r,c);a.fill=r===0?C.navy:(r%2?C.bg:'#EEEEE7');
     a.text.style={typeface:'Arial',fontSize:font*scale,bold:r===0,color:r===0?C.white:C.navy,autoFit:'none'};
   }
 }
 nativeTableSlides.push(presentation.slides.items.length);return t;
}
const P='../../../../Master-mr2/MAJOR_REVISION_PROTOCOL.md';
const F='../../../../Master-mr2/MR3_P3_FINDINGS.md';
const W='../../../../Master-mr2/MAJOR_REVISION_WORKPLAN.md';
const paper='../paper/icomp2026_conference.tex';

// 01. Preserve the original identity and academic style.
{
 const s=slide('',{dark:true});
 text(s,'MinPPI',60,60,1060,97,76,C.mint,true);
 text(s,'Min-Marginal Power Indices for\nMulti-Objective Feature Selection',60,192,1080,130,44,C.white,true);
 text(s,'Research motivation, theoretical basis and experimental design',60,348,1090,52,26,'#D2DEE4');
 line(s,60,438,1080,'#496174',1);
 text(s,'Wenjie SHI',60,469,530,50,29,C.white,true);
 text(s,'Saint Petersburg State University',60,524,720,38,23,C.white);
 text(s,'Applied Mathematics and Control Processes (AMCP)',60,562,910,35,21,C.mint);
 text(s,'Research in progress',815,478,325,40,22,C.mint,false,'right');
 note(s,'建议用时 30 秒。研究主题是 MinPPI：在给定模型、数据划分和效用门槛下寻找特征的上下文有害证据，并研究它能否支持特征选择。本次重点讲研究意义、理论和实验设计，实验结果仍在优化，不把阶段性观察当作最终优势。',[P,W,'Original presentation PDF, slide 1']);
}
// 02. Motivation.
{
 const s=slide('Why study contextual harm?');
 text(s,'A feature can help on average and still break\nan acceptable model in a particular subset.',60,169,1080,110,37,C.navy,true);
 line(s,60,319,1080,C.light,1);
 pair(s,60,354,505,'Scientific motivation','Feature interactions make the value of a variable depend on the other variables in the model.');
 pair(s,645,354,495,'Practical motivation','A concrete failure context can guide a review of feature removal and its cost.');
 text(s,'Potential value: traceable evidence for a selection decision, with utility loss checked separately.',60,561,1065,52,24,C.teal,true);
 note(s,'建议用时 60 秒。先解释为什么“多一个特征”不一定更好：我们讨论的是有限数据和固定学习算法下，重新拟合不同子集的实际表现。平均重要性会概括许多上下文，而研究希望保留一个可具体复查的失败上下文。实际意义是为压缩、后续模型维护提供证据，但是否降低成本或改善预测需要单独实验。这里并不声称相关变量都必须删除，也不把负 SHAP 当成有害标签。',[P+' §1–3',paper+' Introduction']);
}
// 03. Questions and contribution.
{
 const s=slide('Research objective and questions');
 text(s,'Find verifiable harmful contexts under a limited budget,\nthen test whether they support useful removal decisions.',60,166,1080,91,33,C.navy,true);
 row(s,'01',289,'Discovery completeness','How many true harmful features can finite sampling recover?');
 row(s,'02',400,'Decision value','How much compression remains within a prespecified utility tolerance?');
 row(s,'03',511,'Explanation and cost','Which contexts explain the finding, and what does the full procedure cost?');
 note(s,'建议用时 55 秒。项目的目标分为三个可检验的问题。第一，有限预算能找全多少有害见证；第二，删除策略在统一容忍度下是否值得；第三，给出的证据能否复查、代价是否透明。这三个问题互相联系，但第一个见证成立并不能直接推出联合删除安全。预期产出是配额条件下的审计方法、精确参照评价以及独立的删除决策证据。',[P+' §1.1–1.3']);
}
// 04. Theory map.
{
 const s=slide('Theoretical foundations');
 table(s,[
 ['Foundation','Role in this project'],
 ['Cooperative games','Features are players. A coalition is a feature subset.'],
 ['Shapley / Banzhaf','Mean marginal effects motivate a different audit question.'],
 ['Minimum aggregation','MinShap is a direct predecessor. MinPPI uses a binary quota game.'],
 ['Probability and sampling','Witness probability controls finite-budget discovery.'],
 ['Pareto optimization','NSGA-II studies the trade-off between subset size and utility.']
 ],[335,745],{y:177,h:357,font:23});
 text(s,'Contribution focus: quota-conditioned witness semantics and an auditable evaluation procedure.',60,558,1060,52,24,C.teal,true);
 note(s,'建议用时 70 秒。按用途介绍理论，不逐个念名词。合作博弈把特征看作参与者。Shapley 和 Banzhaf 的经典思想是平均边际贡献，而本研究问是否存在负向阈值翻转。最小聚合已有 MinShap 等直接前驱，因此不把 min 本身宣称为首创。本文将其特化为固定效用配额的二值博弈。概率论说明发现能力受见证概率限制；多目标优化用于后续的压缩与效用权衡。SHAP 对模型预测的归因与重新训练后的效用博弈需要区分。',[P+' §2–3','Shapley (1953), A Value for n-Person Games; Banzhaf (1965), Weighted Voting Doesn’t Work: A Mathematical Analysis.','Zheng & Raskutti (2026), MinShap: https://arxiv.org/abs/2604.15107','Lundberg & Lee (2017), SHAP: https://arxiv.org/abs/1705.07874','Deb et al. (2002), NSGA-II: https://doi.org/10.1109/4235.996017']);
}
// 05. Numerical illustration, explicitly invented and labeled.
{
 const s=slide('A concrete witness of harm');
 text(s,'Illustrative values, not an experimental result',60,163,1080,38,22,C.gray);
 text(s,'Full-set utility = 0.90       α = 0.95       Quota q = 0.855',60,217,1090,49,30,C.navy,true);
 table(s,[['Evaluated subset','Utility U','Quota outcome'],['S, before adding feature i','0.87','Pass: v(S) = 1'],['S ∪ {i}, after adding i','0.84','Fail: v(S ∪ {i}) = 0']],[575,175,330],{y:308,h:180,font:26});
 text(s,'Marginal change: 0 − 1 = −1',60,517,1080,55,35,C.red,true);
 text(s,'The two evaluated subsets form a witness that can be checked again.',60,580,1080,37,24,C.navy);
 note(s,'建议用时 65 秒。本页数字仅为教学示例。假设完整特征效用为 0.90，配额为它的 95%，即 0.855。集合 S 的效用 0.87 达标，加上 i 后效用 0.84 不达标，二元边际就是负一。保存特征集合、两侧模型效用和配额就能复查这次负翻转。注意，见证只说明在 S 这个上下文中出现过问题，并没有直接评价从完整集合删除 i 的结果。',[P+' §3','All numerical utilities on this slide are explicitly illustrative.']);
}
// 06. Definition.
{
 const s=slide('MinPPI in a fixed utility game');
 text(s,'q = α U(D)       vα(S) = 1[U(S) ≥ q]',60,167,1080,62,34,C.navy,true);
 text(s,'Mi = min { vα(S ∪ {i}) − vα(S) : S ⊆ D \\ {i} }',60,245,1080,62,32,C.navy,true);
 text(s,'D is the analyzed feature set. U(D) > 0. The default quota is α = 0.95.',60,319,1080,41,22,C.gray);
 text(s,'U: ROC AUC (binary), macro OVR AUC (multiclass), max(R², 0) (regression).',60,361,1080,32,21,C.gray);
 const cols=[{x:60,t:'−1',h:'Harmful',b:'At least one context changes\nfrom pass to fail.',c:C.red},{x:440,t:'0',h:'Neutral at the minimum',b:'No negative context exists,\nbut not every context is positive.',c:C.gray},{x:820,t:'+1',h:'Universally decisive',b:'Every context changes\nfrom fail to pass.',c:C.teal}];
 for(const a of cols){text(s,a.t,a.x,397,320,64,49,a.c,true);text(s,a.h,a.x,464,325,39,25,C.navy,true);text(s,a.b,a.x,516,325,75,22);}
 note(s,'建议用时 70 秒。固定数据划分、预处理、学习器、随机种子、效用和配额，才得到本研究讨论的一个固定博弈。这里 U 是验证集上的表现，训练仍只使用训练集。二元博弈可非单调，所以增加特征可能从达标变成不达标。三个值是精确枚举下的定义，0 并不意味着没有预测价值；+1 要求所有上下文都是正向决定，是非常强的条件。MinSS 与 MinBz 的最小值相同，因为各目标排除子集都能作为某种排列的前驱集合，这不意味着经典加权指数相同。',[P+' §3',paper+' Method / Proposition 1']);
}
// 07. Estimation.
{
 const s=slide('Finite-budget witness search');
 pair(s,60,171,500,'Algorithm A: random contexts','Sample a subset size uniformly, then sample a subset of that size.',{size:26});
 pair(s,645,171,495,'Algorithm B: guided contexts','Build a seed using training mutual information (MI) and correlations, then perturb it.',{size:25});
 line(s,60,372,1080,C.light,1);
 text(s,'Observed −1',60,401,330,46,30,C.red,true);
 text(s,'A checked negative crossing certifies this fixed game.',415,402,710,70,26);
 text(s,'No observed −1',60,497,330,46,30,C.navy,true);
 text(s,'An unvisited context may still contain a harmful witness.',415,498,710,70,26);
 text(s,'Compare A and B at the same total query cap. Guidance is a hypothesis to test.',60,583,1065,39,23,C.teal,true);
 note(s,'建议用时 65 秒。算法 A 并不是对所有子集等概率，而是先等概率抽集合大小，再在该大小内等概率抽子集。算法 B 只使用训练侧的互信息和相关结构，不使用精确标签来指导寻找。采样最小值不低于真实最小值，所以负一一旦出现就有证据，但没有出现负一可能只是漏检。最新协议用总边际查询预算统一 A/B，找到了见证的特征会退出轮转。引导是否更好要看数据，不由理论公式自动保证。',[P+' §4.4–4.6',F+' §5']);
}
// 08. Dataset rationale, the core expansion requested by the user.
{
 const s=slide('Why these six benchmark datasets?');
 table(s,[
 ['Dataset','Samples / inputs¹','Why it is informative'],
 ['Spambase','4,601 / 57','Word statistics in a real binary task'],
 ['Madelon','2,600 / 500','Nonlinear signal, redundancy and 480 probes'],
 ['Gisette','7,000 / 5,000','High dimensional digits with distractor variables'],
 ['ISOLET','7,797 / 617','26-class speech tests multiclass utility'],
 ['Credit-Card-Fraud','284,807 / 29','Large imbalanced task tests cost and robustness'],
 ['Abalone','4,177 / 8','Regression with 8 inputs for exact enumeration']
 ],[235,250,595],{y:168,h:371,font:21});
 text(s,'Selection logic: task diversity, feature structure, scale and an exact reference.',60,553,1080,37,24,C.teal,true);
 text(s,'¹ Project inputs. Madelon/Gisette use labeled train + validation data. Fraud excludes Time.',60,599,1080,30,17,C.gray);
 note(s,'建议用时 100 秒。这是重点页。数据集选择不是单纯堆数量，而是覆盖不同可检验条件：Spambase 的词频统计便于理解真实特征；Madelon 有已知非线性结构、冗余和大量干扰列，适合检验发现能力；Gisette 检验高维计算压力；ISOLET 覆盖 26 分类；Fraud 检验严重不平衡和大样本成本；Abalone 用八个原始变量提供回归与精确枚举参照。这里的“为什么有意义”是基于数据属性的研究设计解释，不是已经观察到算法优势。Madelon 2,600 和 Gisette 7,000 是项目合并有标签 train/validation 的样本数，不是官网包含无标签测试集的总数。Fraud 项目使用 V1–V28 和 Amount，排除 Time。',[
 paper+' Table: Datasets','../../configs/credit_card_fraud.yaml','https://archive.ics.uci.edu/dataset/94/spambase','https://archive.ics.uci.edu/dataset/171/madelon','https://archive.ics.uci.edu/dataset/170/gisette','https://archive.ics.uci.edu/dataset/54/isolet','https://archive.ics.uci.edu/dataset/1/abalone','https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud']);
}
// 09. Scope and integrity.
{
 const s=slide('Two evaluation scales, with explicit boundaries');
 pair(s,60,166,510,'Broad benchmark scope','Six datasets test different tasks and dimensions.\nTraining-only MI limits large feature pools to at most 100 columns.',{size:25});
 pair(s,645,166,495,'Exact reference scope','Four datasets provide fixed 8-feature games.\nAll 256 subsets are feasible to evaluate.',{size:25});
 line(s,60,385,1080,C.light,1);
 text(s,'Why smaller games?',60,411,405,43,29,C.teal,true);
 text(s,'Exact labels make missed witnesses measurable. Labels apply to the selected panel, not the original high dimensional problem.',475,409,665,95,25);
 text(s,'Data integrity',60,538,405,43,29,C.teal,true);
 text(s,'Fit preprocessing on training data only. Exclude Communities & Crime after the target-leakage audit.',475,536,665,75,25);
 note(s,'建议用时 65 秒。必须区分两个层次：旧六数据集实验覆盖不同任务；最新精确发现实验只使用四个数据集的八维游戏，便于枚举 256 个子集。降维后的精确标签只属于所定义的面板，不能宣称得到 100 维全局真值。两个非 Abalone 面板规则是 MI top8 和 MI top4+random4，均只看训练数据；Abalone 保留全部八列。新八维参照只去掉训练方差为零的列，与旧正阈值筛选有协议区别。Communities & Crime 的目标可由保留的结果成分列重构，所以已从发表范围移除；不能用泄漏产生的高分支撑研究。',[P+' §4.1',paper+' Experimental design','../../README.md']);
}
// 10. E1.
{
 const s=slide('Experiment 1: discovery against exact references');
 text(s,'35 fixed games',60,170,525,67,45,C.teal,true);
 text(s,'4 datasets, 7 panels, 5 data splits',60,243,550,43,25);
 text(s,'256 subset utilities per game',60,295,565,48,29,C.navy,true);
 text(s,'Enumerate all subsets and retain every negative witness.',60,354,540,82,25);
 text(s,'Fair A/B comparison',675,180,465,49,30,C.navy,true);
 text(s,'30 paired sampling schedules per game\nTotal marginal-query cap Q = 800\nSampler sees only its requested utilities',675,249,455,151,25);
 line(s,60,466,1080,C.light,1);
 text(s,'Primary endpoint',60,488,385,44,28,C.teal,true);
 text(s,'Recall across the full query budget, summarized by recall area.',460,488,680,74,26);
 text(s,'Empty harmful sets remain in coverage reports. Their recall is undefined.',60,583,1080,36,23,C.gray);
 note(s,'建议用时 75 秒。E1 的 35 个游戏已经完成，这是当前最直接的发现能力评价。组成是 Abalone 1 面板、其余三个数据集各 2 面板，再乘 5 个划分。每个八维游戏完整计算 256 个效用，之后 A/B 通过盲查询接口采样，不能访问完整表或精确标签。每个游戏 30 次日程重复，总上限 800 个边际查询。主终点为每次查询后的召回率平均值，也就是归一化召回面积。没有 harmful 的游戏不是失败，召回应记 NA，并保留分母。30 次日程重复也不是 30 个独立数据集。',[P+' §4',F+' §2–4']);
}
// 11. E2.
{
 const s=slide('Planned experiment 2: controlled mechanisms');
 text(s,'15 prespecified instances: 3 mechanisms × 5 instances',60,169,1080,49,31,C.navy,true);
 table(s,[
 ['Mechanism','Construction','Question it isolates'],
 ['Additive signal','Two signal variables + six irrelevant variables','Does irrelevant input produce harmful contexts?'],
 ['Correlated redundancy','Two signal variables + two noisy proxies + four irrelevant variables','How does redundancy change witness discovery?'],
 ['Interaction','Noisy XOR of two variables + six irrelevant variables','Can sampling recover contexts driven by interactions?']
 ],[250,425,405],{y:244,h:283,font:22});
 text(s,'Each instance: 2,000 rows, 8 features, full enumeration and the same A/B budget.',60,549,1080,43,24,C.teal,true);
 text(s,'A variable’s generating role does not determine its MinPPI label.',60,603,1080,29,20,C.gray);
 note(s,'建议用时 75 秒。E2 尚未实施。真实数据很难解释哪些结构导致漏检，所以补三个受控机制。S1 是 logistic(x1+x2) 的二元任务；S2 额外加入 x3=x1+0.1e3、x4=x2+0.1e4；S3 是两变量符号的 XOR，再独立翻转 10% 的标签。每种五个固定实例，各 2,000 行八列。仍用同一模型和完整枚举，避免把生成角色当成审计真值：生成上无关不等于固定样本、模型、配额下有害。所有 15 个槽位都保留，不根据效果更换 seed。',[P+' §22.3 and §23','../../../../Master-mr2/MR3_E2_PLAN.md']);
}
// 12. E4.
{
 const s=slide('Planned experiment 3: removal decisions');
 table(s,[
 ['Data role','Share','Permitted use'],
 ['Train','64%','Fit preprocessing, models and feature rankings'],
 ['Selection validation','8%','Audit contexts and choose one candidate per strategy'],
 ['Independent gate','8%','Accept the candidate or revert to the full set'],
 ['Held-out test','20%','Evaluate the frozen decision after refitting']
 ],[310,140,630],{y:170,h:310,font:24});
 text(s,'Gate rule: candidate loss ≤ 0.005 AUC or ≤ 0.01 raw R²',60,506,1080,53,30,C.teal,true);
 text(s,'A verified local witness alone does not establish that joint removal is useful.',60,579,1080,40,24,C.navy);
 note(s,'建议用时 80 秒。这是应用层面最关键的改进，而且尚未完成。旧方案反复用同一验证集寻找见证和做 gate，新 E4 把验证部分分成独立的选择集与 gate。每种策略只能提交一个候选，gate 不通过就回 full，不能继续尝试直到通过。子集冻结后，统一在非测试 80% 上重新拟合模型，预处理参数仍来自原 64% train。分类允许损失 0.005 AUC，回归 0.01 raw R²。这只是预先固定的有限样本点估计规则，不等于统计上的无损或非劣保证。首批为四数据集各五划分，共 20 个八维决策游戏。',[P+' §7 and §22.4']);
}
// 13. Baselines/metrics.
{
 const s=slide('Practical baselines and success criteria');
 text(s,'Primary comparison: MinPPI-A versus a validation-only portfolio',60,171,1080,78,32,C.navy,true);
 pair(s,60,270,505,'Eight strategies','Full, MinPPI-A, MinPPI-B, MI, TreeSHAP, static LOFO, RFE and the portfolio.',{size:25});
 pair(s,645,270,495,'A demanding practical control','The portfolio selects among MI, TreeSHAP and LOFO candidates using selection validation only.',{size:25});
 line(s,60,446,1080,C.light,1);
 text(s,'Measure decision value',60,470,415,43,28,C.teal,true);
 text(s,'Compression, test utility loss, tolerance violations and gate rejection.',480,470,660,76,25);
 text(s,'Measure actual cost',60,565,415,43,28,C.teal,true);
 text(s,'Count queries and model fits. Include guidance, selection, gate and refit time.',480,562,660,65,24);
 note(s,'建议用时 65 秒。不能只赢过随机删除就认为有实用价值，因此引入简单、可解释的强对照。MI 按互信息，TreeSHAP 按平均绝对归因，静态 LOFO 逐一去掉特征后只计算一次得分，RFE 是同一 XGBoost 上的递归特征消除。portfolio 在选择验证集上合并三种排名的可行前缀，取最大压缩，并计入三个排名的全成本。主比较预先固定为 A 对 portfolio，不能看到 B 的效果后换主比较。逐数据集报告测试效用差，不将 AUC 和 R² 的原始差值直接平均成总体结论。',[P+' §7.3–7.4 and §22.4']);
}
// 14. Secondary objective theory.
{
 const s=slide('Multi-objective feature selection');
 text(s,'Two competing objectives',60,172,1080,47,32,C.navy,true);
 text(s,'Minimize the retained feature fraction:  |S| / p₀',60,244,1080,57,31,C.teal,true);
 text(s,'Minimize predictive loss:  1 − clip(U(S), 0, 1)',60,319,1080,57,31,C.teal,true);
 line(s,60,409,1080,C.light,1);
 pair(s,60,440,510,'NSGA-II search','A nondominated set offers different compression–utility trade-offs.',{size:24});
 pair(s,645,440,495,'MinPPI as a possible prefilter','Compare search quality and full cost, including the preceding audit.',{size:24});
 note(s,'建议用时 60 秒。题目中的多目标特征选择由两个目标构成：保留特征比例和预测损失。一个解只有在至少一项更好、另一项不更差时才支配另一个解，所以最终得到多个折中解。NSGA-II 通过非支配排序和拥挤距离保持这些解的多样性。MinPPI 可在搜索前给出候选过滤，但这是次要应用。p0 必须使用过滤前相同的维数分母。旧实验的种群是 40、每个变体 800 evaluations，超体积参照点 (1.1,1.1)。相同搜索评价次数不代表同端到端耗时，重新比较前要把审计成本纳入。最新路线先完成发现和独立删除实验，再决定是否扩展搜索。',[paper+' Experimental design',P+' §7','Deb et al. (2002), https://doi.org/10.1109/4235.996017']);
}
// 15. One page only for current results.
{
 const s=slide('Current evidence and remaining experiments',{tag:'Project evidence status: 26 September 2026'});
 table(s,[
 ['Evidence layer','Status','What it currently supports'],
 ['Exact reference audit','35 games complete','Harmful set: 22 nonempty, 13 empty.'],
 ['A/B discovery comparison','Complete','Recall area A/B: 0.4012 / 0.3718'],
 ['Cold-cache checks','Six checks complete','Queried utilities and traces reproduced.'],
 ['Controlled mechanisms','Planned','15 instances to examine mechanisms.'],
 ['Deletion decisions','Planned','20 games to test deletion value and cost.']
 ],[310,245,525],{y:173,h:350,font:22});
 text(s,'B has no stable advantage. Practical removal value remains under evaluation.',60,550,1060,61,28,C.teal,true);
 note(s,'建议用时 45 秒。按用户要求结果简要带过。以 9 月 26 日最新台账为准：35 个游戏完整完成，其中 22 个有 harmful，13 个为空；A/B 宏平均召回面积为 0.4012/0.3718，不能宣称 B 更优。该宏平均先在同游戏平均 30 日程，再到面板、数据集，最后四数据集等权，仅有定义游戏进入召回，空游戏仍保留覆盖分母。冷缓存六次检查只表明已访问的效用及轨迹可重现，不证明稳定加速。旧六数据集结果也不建立普遍选择优势。下一步的 E2 和独立 gate/test 决策实验仍未完成。',[F+' §1–6',W+' Latest recovery entry']);
}
// 16. Close.
{
 const s=slide('Expected contribution and next steps',{dark:true});
 text(s,'A feature audit that explains where a variable\nbreaks a defined utility requirement.',60,182,1070,120,41,C.white,true);
 text(s,'Exact references quantify what the audit misses.\nIndependent decisions test whether removal is worth the cost.',60,341,1080,108,29,C.mint);
 line(s,60,495,1080,'#496174',1);
 text(s,'Next evidence',60,529,340,42,28,C.mint,true);
 text(s,'Controlled mechanisms, followed by independent gate and test evaluation.',430,528,710,83,28,C.white);
 note(s,'建议用时 40 秒。结尾回到意义：提供一个可以复查的负向阈值翻转上下文，让特征选择的理由更具体；用精确参照衡量遗漏；再独立评价删除收益与成本。预期贡献不依赖必须证明 B 胜过 A，也不承诺通用选择优势。接下来按固定计划完成合成机制和独立决策实验。主讲至此结束，附录只在提问时使用。',[P+' §1 and §22',F+' Next steps']);
}
// 17. Backup theory.
{
 const s=slide('Backup: exact and sampled guarantees');
 row(s,'01',167,'Support-level equality','Every target-excluding subset can precede that feature in a permutation.');
 text(s,'MinSSᵢ = MinBzᵢ = Mᵢ',145,270,990,51,31,C.teal,true);
 row(s,'02',345,'Sample minimum is an upper bound','A sample visits only part of the domain. An observed −1 is conclusive.');
 text(s,'If a fixed witness has probability ρ > 0 in K independent draws:',60,499,1080,41,25);
 text(s,'P(detect a harmful witness) ≥ 1 − (1 − ρ)ᴷ',60,550,1080,60,33,C.teal,true);
 note(s,'备用页。第一个结论只针对 support-level minimum，不针对经典带权平均 Shapley–Shubik 与 Banzhaf 指数。对于第二个结论，有限采样子集上的最小值大于等于全域最小值。若固定指导条件下，某个负见证每次被抽到的概率是 rho，独立 K 次至少命中它的概率为 1-(1-rho)^K，因此发现任意负见证的概率至少是这个值。若 rho=0 就没有正的发现保证。条件概率公式不保证引导 B 的 rho 高于随机 A，也不能把自适应全局轮转预算 Q 直接替代局部 IID 次数 K。',[P+' §3–4',F+' §5',paper+' Proposition 1 and finite-budget estimation']);
}
// 18. Backup bibliography and provenance.
{
 const s=slide('Backup: sources and protocol details');
 text(s,'Theory',60,171,240,43,28,C.teal,true);
 text(s,'Shapley (1953), A Value for n-Person Games\nBanzhaf (1965), Weighted Voting Doesn’t Work\nLundberg & Lee (2017), A Unified Approach to Interpreting Model Predictions\nZheng & Raskutti (2026), MinShap: A Shapley-Based Framework for Feature Redundancy\nDeb et al. (2002), A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II',310,170,830,235,21);
 line(s,60,432,1080,C.light,1);
 text(s,'Data and protocol',60,456,240,43,28,C.teal,true);
 text(s,'UCI dataset documentation and the ULB credit-card dataset\nLocal revision protocol and the 35-game P3 findings, 26 September 2026\nXGBoost: 100 rounds, depth 4, learning rate 0.1\nFive split identifiers: 42, 123, 456, 789, 1024',310,455,830,153,22);
 note(s,'备用页。正文数据集属性参考原始发布方；实验数字和状态参考当前项目文件。精确参照游戏使用 64/16/20 划分但不访问 test；计划中的决策实验为 64/8/8/20，两者不是同一个效用博弈。新审计中回归效用为 max(raw R²,0)，同时保存 raw R²，删除 gate 和测试评价使用 raw R²。全部引用和每页中文提示在备注中。',[P,W,F,paper,'https://arxiv.org/abs/1705.07874','https://arxiv.org/abs/2604.15107','https://doi.org/10.1109/4235.996017','https://archive.ics.uci.edu/dataset/94/spambase','https://archive.ics.uci.edu/dataset/171/madelon','https://archive.ics.uci.edu/dataset/170/gisette','https://archive.ics.uci.edu/dataset/54/isolet','https://archive.ics.uci.edu/dataset/1/abalone','https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud']);
}

await fs.mkdir(tmp,{recursive:true});
await fs.mkdir(path.join(workspaceDir,'revised'),{recursive:true});
const candidatePath=path.join(tmp,'candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
console.log('EXPORTED',candidatePath);
for(let i=0;i<presentation.slides.items.length;i++){
 const s=presentation.slides.items[i];
 const png=await presentation.export({slide:s,format:'png',scale:1});
 await fs.writeFile(path.join(tmp,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
 const lay=await s.export({format:'layout'});
 await fs.writeFile(path.join(tmp,`slide-${String(i+1).padStart(2,'0')}.json`),await lay.text());
 console.log('RENDERED',i+1);
}
await fs.writeFile(path.join(tmp,'speaker-notes.txt'),notes.map((v,i)=>`SLIDE ${i+1}\n${v}`).join('\n\n'));
const result=await finalizePresentation({
 workspaceDir,candidatePath,finalPath:path.join(workspaceDir,'revised','MinPPI_Defense_Wenjie_SHI_15min.pptx'),
 pythonExecutable:RUNTIME_PYTHON,
 integrityValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','15240000,8572500','--validate-bullet-geometry','--validate-heading-fit',...nativeTableSlides.flatMap(n=>['--require-native-table-slide',String(n)])],
 explicitTotalSlideCount:18,
 requiredNativeTableOwnerSlides:nativeTableSlides,
 requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'design',families:['Arial'],scriptFonts:{ea:'Microsoft YaHei'}},
 verifyArtifactToolImport:true,
 receiptPath:path.join(tmp,'validation-15min.json')
});
console.log(JSON.stringify(result));
const finalDeck=await PresentationFile.importPptx(await FileBlob.load(result.finalPath));
for(let i=0;i<finalDeck.slides.items.length;i++){
 const png=await finalDeck.export({slide:finalDeck.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(tmp,`final-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
console.log('FINAL_RENDERED',finalDeck.slides.items.length);
