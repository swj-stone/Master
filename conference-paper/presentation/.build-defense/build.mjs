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
 const timings=[25,40,35,60,40,45,50,45,50,85,85,55,50,65,50,40,40,30];
 if(notes.length<18) cn=cn.replace(/建议用时 \d+ 秒/,`建议用时 ${timings[notes.length]} 秒`);
 const root='C:/work/学校/A.SPBU/paper/ZhaoCHI/Paper/';
 sources=sources.map(v=>v.replace('../../../../Master-mr2/',root+'Master-mr2/').replace('../../configs/',root+'Master/configs/').replace('../../README.md',root+'Master/README.md').replace('../paper/',root+'Master/conference-paper/paper/'));
 const n=`讲述提示\n${cn}\n\n资料来源\n${sources.join('\n')}`;s.speakerNotes.textFrame.setText(n);notes.push(n);
}
function pair(s,x,y,w,head,body,{color=C.teal,size=25}={}){text(s,head,x,y,w,40,28,color,true);text(s,body,x,y+55,w,135,size);}
function row(s,n,y,head,body){text(s,n,60,y,70,45,28,C.teal,true);text(s,head,145,y,985,43,28,C.navy,true);text(s,body,145,y+49,985,71,24,C.gray);}
function table(s,values,widths,{x=60,y=190,h=340,font=21,rowHeights=null}={}){
 const w=widths.reduce((a,b)=>a+b,0);
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x*scale,top:y*scale,width:w*scale,height:h*scale,columnWidths:widths.map(v=>v*scale),values});
 t.borders.assign({fill:C.light,width:0.7*scale,style:'solid'});
 for(let r=0;r<values.length;r++){
   t.rows[r].height=(rowHeights?rowHeights[r]:h/values.length)*scale;
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
 text(s,'Supervisor: Chi ZHAO',730,477,410,42,26,C.white,true,'right');
 text(s,'Research in progress',815,521,325,37,21,C.mint,false,'right');
 note(s,'建议用时 30 秒。研究主题是 MinPPI：在给定模型、数据划分和效用门槛下寻找特征的上下文有害证据，并研究它能否支持特征选择。本次重点讲研究意义、理论和实验设计，实验结果仍在优化，不把阶段性观察当作最终优势。',[P,W,'Original presentation PDF, slide 1']);
}
// 02. Motivation.
{
 const s=slide('Why study contextual harm?');
 text(s,'A feature can help on average and still break\nan acceptable model in a particular subset.',60,169,1080,110,37,C.navy,true);
 line(s,60,319,1080,C.light,1);
 pair(s,60,346,505,'Scientific motivation','Redundancy, interactions and finite-sample fitting can make a variable useful in one subset and harmful in another.',{size:25});
 pair(s,645,346,495,'Practical motivation','Fewer input variables may reduce measurement and model cost. A checked failure context gives a concrete reason to test removal.',{size:25});
 text(s,'Potential value: traceable evidence for a selection decision, with utility loss checked separately.',60,561,1065,52,24,C.teal,true);
 note(s,'建议用时 60 秒。先解释为什么“多一个特征”不一定更好：我们讨论的是有限数据和固定学习算法下，重新拟合不同子集的实际表现。平均重要性会概括许多上下文，而研究希望保留一个可具体复查的失败上下文。实际意义是为压缩、后续模型维护提供证据，但是否降低成本或改善预测需要单独实验。这里并不声称相关变量都必须删除，也不把负 SHAP 当成有害标签。',[P+' §1–3',paper+' Introduction']);
}
// 03. Questions and contribution.
{
 const s=slide('Research objective and questions');
 text(s,'Find verifiable harmful contexts under a limited budget,\nthen test whether they support useful removal decisions.',60,166,1080,91,33,C.navy,true);
 row(s,'01',289,'Discovery completeness','Exact small-game labels measure recall and missed witnesses at a fixed query budget.');
 row(s,'02',400,'Decision value','Compare compression and test loss with MI, SHAP, LOFO, RFE and a portfolio.');
 row(s,'03',511,'Explanation and cost','Record the subset, both utilities and quota. Count queries, fits and total time.');
 note(s,'建议用时 55 秒。项目的目标分为三个可检验的问题。第一，有限预算能找全多少有害见证；第二，删除策略在统一容忍度下是否值得；第三，给出的证据能否复查、代价是否透明。这三个问题互相联系，但第一个见证成立并不能直接推出联合删除安全。预期产出是配额条件下的审计方法、精确参照评价以及独立的删除决策证据。',[P+' §1.1–1.3']);
}
// Learning paradigm and the meaning of an input feature.
{
 const s=slide('Learning setting: supervised feature selection');
 text(s,'Each observation has input variables X and a known target y.',60,165,1080,50,31,C.navy,true);
 table(s,[
 ['Object','Meaning in this project','Example: an email'],
 ['Sample / row','One labeled observation','One email with its spam label'],
 ['Feature / column xᵢ','A predictor available to the model','word_freq_free or char_freq_!'],
 ['Target y','The outcome to predict, excluded from X','is_spam: 1 or 0'],
 ['Coalition S','A selected group of input columns','Two word frequencies + capital count']
 ],[220,445,415],{y:232,h:278,font:21});
 text(s,'For each S: fit XGBoost on labeled training rows, then compute U(S) on validation rows.',60,533,1080,58,25,C.teal,true);
 text(s,'Classification and regression. No clustering objective or agent–environment reward loop.',60,602,1080,32,20,C.gray);
 note(s,'建议用时 60 秒。这一页明确学习范式。我们手里有成对的输入 X 和标签 y，模型根据已知标签学习映射，所以六个任务都是监督学习。特征是作为预测输入的一列，不是一个样本，也不是目标标签。以邮件为例，每一行是一封邮件，free 的词频、感叹号频率是输入列，是否垃圾邮件是目标。MinPPI 中的参与者就是当前表示的一列，联盟是一组列。每选一个联盟，都用相同的监督学习模型重新训练，再通过验证标签计算效用。数据集可以被用于其他研究，但当前项目并未执行无监督聚类或强化学习。PCA 是部分数据的表示来源，也不会改变这里的监督分类任务。',[P+' §3','../../configs/spambase.yaml','C:/work/学校/A.SPBU/paper/ZhaoCHI/Paper/Master/experiments/preprocess_data/preprocess_Spambase.py']);
}
// Theory map.
{
 const s=slide('Theoretical foundations');
 table(s,[
 ['Foundation','Role in this project'],
 ['Cooperative games','Features are players. A coalition is a feature subset.'],
 ['Shapley / Banzhaf','Average marginal effects with different subset weights.'],
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
// Dataset task matrix.
{
 const s=slide('Dataset tasks, targets and evaluation metrics');
 text(s,'All six datasets serve supervised learning in this project.',60,159,1080,46,29,C.teal,true);
 table(s,[
 ['Dataset','Rows / inputs¹','Target y','Task','Metric'],
 ['Spambase','4,601 / 57','Spam / non-spam','Binary','ROC AUC'],
 ['Madelon','2,600 / 500','Synthetic class ±1','Binary','ROC AUC'],
 ['Gisette','7,000 / 5,000','Digit 4 / 9','Binary','ROC AUC'],
 ['ISOLET','7,797 / 617','Spoken letter A–Z','26 classes','OVR AUC'],
 ['Credit-card fraud','284,807 / 29','Fraud: 1 / 0','Binary','ROC AUC'],
 ['Abalone','4,177 / 8','Rings count','Regression','R²']
 ],[215,195,300,185,185],{y:219,h:344,font:20});
 text(s,'¹ Project input pools. Madelon/Gisette use labeled train + validation rows. Fraud excludes Time.',60,578,1080,32,18,C.gray);
 text(s,'Train-only preprocessing caps large pools at 100 inputs. A leakage audit excluded Communities & Crime.',60,613,1060,27,17,C.gray);
 note(s,'建议用时 50 秒。数据集中有标签，使我们能够把模型预测与真实目标进行比较，从而定义每个子集的效用。四个二分类使用 ROC AUC，ISOLET 是 26 分类并使用宏平均 one-vs-rest AUC，Abalone 在项目中将 Rings 作为回归目标并用 R² 评价。Madelon 虽然是人工生成数据，仍有标签，因而是监督分类。Abalone 的 Rings 与年龄有关，但本项目直接预测环数，不把年龄另加为输入。行数是项目使用的有标签样本量，列数不含目标。较高维池还会进行训练侧 MI 预筛选到最多100列。',[
 paper+' Table: Datasets','../../configs/credit_card_fraud.yaml','https://archive.ics.uci.edu/dataset/94/spambase','https://archive.ics.uci.edu/dataset/171/madelon','https://archive.ics.uci.edu/dataset/170/gisette','https://archive.ics.uci.edu/dataset/54/isolet','https://archive.ics.uci.edu/dataset/1/abalone','https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud']);
}
// Actual variable composition and the role of each dataset.
{
 const s=slide('Input features and their role in MinPPI (1/2)');
 table(s,[
 ['Dataset','What the input columns represent','Why this structure fits the study'],
 ['Spambase\n57 inputs','48 word frequencies, e.g. free, money\n6 character frequencies, e.g. !, $\n3 capital-run statistics: mean, max, total','Observable text statistics make subset witnesses easier to interpret. Real binary data complements synthetic structure.'],
 ['Madelon\n500 inputs','5 informative coordinates\n15 linear combinations of them\n480 irrelevant probe variables','Known redundancy and distractors test discovery under nonlinear interactions. Generating roles are not MinPPI labels.'],
 ['Gisette\n5,000 inputs','2,500 image-derived variables, including pixel products\n2,500 distractor probes\nNumeric columns encode digit images','High dimensional, nonlinear visual features stress screening and search cost. The project analyzes a train-selected pool.']
 ],[185,480,415],{y:173,h:408,font:21,rowHeights:[45,121,121,121]});
 text(s,'A MinPPI player is an analyzed column. A coalition selects columns while retaining the same target.',60,599,1080,31,20,C.teal,true);
 note(s,'建议用时 85 秒。详细说明三类变量。Spambase 的48维词频指某词占邮件单词数的百分比，6维字符频率和3维连续大写统计共同构成57列。目标 is_spam 不属于特征。Madelon 的5个信息维度加15个线性组合构成20个相关信息变量，另480个干扰列，总计500列。Gisette 把手写4和9图像转成数值描述，含像素及高阶乘积等衍生变量，再加入2500个干扰变量。这里的特征既可能是可解释的原始统计量，也可能是变换后的数值坐标。Madelon、Gisette 的原始变量身份并未按语义逐列公开，不能给 f_i 编造实际含义。已知生成角色也不能替代固定模型下的精确有害标签。',[P+' §3–4','https://archive.ics.uci.edu/dataset/94/spambase','https://archive.ics.uci.edu/dataset/171/madelon','https://archive.ics.uci.edu/dataset/170/gisette','C:/work/学校/A.SPBU/paper/ZhaoCHI/Paper/Master/experiments/preprocess_data/preprocess_Spambase.py']);
}
{
 const s=slide('Input features and their role in MinPPI (2/2)');
 table(s,[
 ['Dataset','What the input columns represent','Why this structure fits the study'],
 ['ISOLET\n617 inputs','Acoustic descriptors: spectral coefficients, contours, sonorant and surrounding-segment features\nTarget: the spoken letter','Tests whether witness auditing extends to multiclass utility. Individual column order has no published semantic map.'],
 ['Credit-card fraud\n29 inputs','28 anonymized PCA components V1–V28\nAmount: transaction value\nTime is excluded from project inputs','Large labeled data with rare fraud cases tests AUC-based decisions and cost. Witnesses concern components, not raw financial attributes.'],
 ['Abalone\n8 inputs','Sex; Length; Diameter; Height\nWhole weight; Shucked weight\nViscera weight; Shell weight\nTarget: Rings, excluded from inputs','Physical measurements provide a regression task. Eight variables permit all 256 subsets and exact harmful labels.']
 ],[185,480,415],{y:173,h:408,font:21,rowHeights:[45,121,121,121]});
 text(s,'Labeled rows allow retraining on X[S] and measuring U(S). Harm depends on the model, split and quota.',60,599,1080,31,20,C.teal,true);
 note(s,'建议用时 85 秒。ISOLET 的617个数值是已经提取好的声学描述，不是原始音频波形；原始资料没有给出每列的具体语义顺序，所以只说明可核实的特征族。Fraud 是监督二分类，V1到V28是匿名PCA主成分，加上金额构成项目的29列，Time不进入本项目输入。PCA作为表示处理不意味着当前任务是无监督学习，标签 Class 仍用于训练和效用评价。PCA成分的负见证也不能直接解释成某个原始金融行为有害。Abalone 的八列全部列出，其中Sex是原始类别变量在现有数据中的数值表示，其余七列是尺寸和重量。Rings是目标，不是第九个输入。完整八维枚举提供精确参照。这些数据能用于项目的共同原因是标签、可选择的有限列集合，以及可计算的监督预测效用。',[P+' §3–4','https://archive.ics.uci.edu/dataset/54/isolet','https://www.openml.org/api/v1/json/data/1597','https://archive-beta.ics.uci.edu/dataset/1/abalone','C:/work/学校/A.SPBU/paper/ZhaoCHI/Paper/Master/data/raw/abalone.csv','C:/work/学校/A.SPBU/paper/ZhaoCHI/Paper/Master/data/raw/credit_card_fraud.csv']);
}
// 10. E1.
{
 const s=slide('Experiment 1: discovery against exact references');
 text(s,'35 fixed games',60,170,525,67,45,C.teal,true);
 text(s,'Abalone, Spambase, Madelon, credit-card fraud\n7 panels × 5 fixed data splits',60,232,550,65,22);
 text(s,'256 subset utilities per game',60,295,565,48,29,C.navy,true);
 text(s,'Abalone: all 8 inputs. Other datasets: MI top8 and MI top4 + random4 panels.',60,351,550,88,24);
 text(s,'Fair A/B comparison',675,180,465,49,30,C.navy,true);
 text(s,'30 paired sampling schedules per game\nTotal marginal-query cap Q = 800\nSampler sees only its requested utilities',675,249,455,151,25);
 line(s,60,466,1080,C.light,1);
 text(s,'Primary endpoint',60,488,385,44,28,C.teal,true);
 text(s,'Recall across the full query budget, summarized by recall area.',460,488,680,74,26);
 text(s,'Exact labels belong to each 8-feature panel. Empty harmful sets have undefined recall.',60,583,1080,36,22,C.gray);
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
 workspaceDir,candidatePath,finalPath:path.join(workspaceDir,'revised','MinPPI_Defense_Wenjie_SHI_15min_Expanded.pptx'),
 pythonExecutable:RUNTIME_PYTHON,
 integrityValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','15240000,8572500','--validate-bullet-geometry','--validate-heading-fit',...nativeTableSlides.flatMap(n=>['--require-native-table-slide',String(n)])],
 explicitTotalSlideCount:20,
 requiredNativeTableOwnerSlides:nativeTableSlides,
 requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'design',families:['Arial'],scriptFonts:{ea:'Microsoft YaHei'}},
 verifyArtifactToolImport:true,
 receiptPath:path.join(tmp,'validation-expanded.json')
});
console.log(JSON.stringify(result));
const finalDeck=await PresentationFile.importPptx(await FileBlob.load(result.finalPath));
for(let i=0;i<finalDeck.slides.items.length;i++){
 const png=await finalDeck.export({slide:finalDeck.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(tmp,`final-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
console.log('FINAL_RENDERED',finalDeck.slides.items.length);
