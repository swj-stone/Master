import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile, FileBlob } from '@oai/artifact-tool';

const workspaceDir = 'C:/work/学校/A.SPBU/paper/ZhaoCHI/Paper/Master/conference-paper/presentation';
const tmp = path.join(workspaceDir, '.build-defense', 'main-only');
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
 const timings=[30,60,55,70,70,80,60,60,85,45,60,55,80,45,45];
 if(notes.length<15) cn=cn.replace(/建议用时 \d+ 秒/,`建议用时 ${timings[notes.length]} 秒`);
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

// Native editable diagrams show the requested research architecture.
function node(s,x,y,w,h,head,body,{accent=false,size=23}={}) {
 const a=s.shapes.add({geometry:'rect',name:`stage-${++count}`,position:{left:x*scale,top:y*scale,width:w*scale,height:h*scale},fill:accent?'#E0F0EB':C.bg,line:{fill:accent?C.teal:C.gray,width:1.3*scale}});
 text(s,head,x+16,y+12,w-32,43,25,accent?C.teal:C.navy,true);
 if(body) text(s,body,x+16,y+58,w-32,h-69,size,C.navy);
 return a;
}
function connect(s,a,b,from='right',to='left',route='straight') {s.shapes.connect(a,b,{kind:route,fromSide:from,toSide:to,line:{fill:C.teal,width:2*scale},tail:{type:'triangle',width:'med',length:'med'}});}
function claim(s,value,y=573){text(s,value,60,y,1080,60,26,C.teal,true);}

// 01. Identity and one plain-language research question.
{
 const s=slide('',{dark:true});
 text(s,'MinPPI',60,59,1080,75,48,C.mint,true);
 text(s,'Min-Marginal Power Indices for\nMulti-Objective Feature Selection',60,173,1080,150,44,C.white,true);
 text(s,'When can an extra feature make a model worse?',60,351,1080,65,27,C.mint);
 line(s,60,451,1080,'#496174',1);
 text(s,'Wenjie SHI',60,475,480,47,29,C.white,true);
 text(s,'Supervisor: Chi ZHAO',670,475,470,47,26,C.white,true,'right');
 text(s,'Saint Petersburg State University\nApplied Mathematics and Control Processes',60,539,800,74,22,C.mint);
 note(s,'建议用时 30 秒。开场不念长标题。我的研究关注一个具体问题：我们往模型里增加一个输入变量时，它是否可能让原本达到要求的模型反而达不到要求？我希望保存这种失败的具体证据，再检验它能否帮助我们选择更小的特征集合。本次重点介绍研究设计，实验仍在推进。',[P,paper]);
}
// 02. Motivation through one familiar application.
{
 const s=slide('Why feature selection needs evidence');
 text(s,'Example: a spam classifier uses many measurements of an email.',60,161,1080,62,30,C.navy,true);
 const a=node(s,60,253,310,137,'Input measurements','Word frequencies\nPunctuation and capitals');
 const b=node(s,452,253,290,137,'Train a model','Learn from emails\nwith known labels');
 const c=node(s,825,253,315,137,'Predict','Spam or non-spam');
 connect(s,a,b);connect(s,b,c);
 text(s,'Why reduce inputs?',60,446,370,45,28,C.teal,true);
 text(s,'Less measurement and model cost. Fewer redundant inputs.\nBut a smaller model must still meet the prediction requirement.',440,437,700,100,26);
 claim(s,'The research need: an inspectable reason to test a feature for removal.');
 note(s,'建议用时 60 秒。用垃圾邮件解释输入、模型和目标。每封邮件是一行数据，词频、标点、大写统计是输入，是否垃圾邮件是标签。实际使用时我们希望减少输入采集和维护成本。但固定学习器在有限样本下训练时，增加或删除输入的效果取决于其他输入，不能只凭直觉。减少特征能否节省总成本仍需要把审计开销一起测量。本研究希望为删除测试提供可复查的理由。',[paper+' Introduction','https://archive.ics.uci.edu/dataset/94/spambase',P+' §1']);
}
// 03. Two complementary checks with different conclusions.
{
 const s=slide('Two checks: addition and removal');
 text(s,'First find a harmful context.\nThen test whether removal helps.',60,169,1080,108,37,C.navy,true);
 text(s,'1. Addition check',60,307,515,45,29,C.teal,true);
 text(s,'2. Removal check',650,307,490,45,29,C.teal,true);
 text(s,'Compare S with S plus input i.',60,365,515,62,26,C.navy,true);
 text(s,'Compare full inputs with a reduced set.',650,365,490,62,26,C.navy,true);
 text(s,'Does adding i make this feature group fall below the performance requirement?',60,441,515,112,26);
 text(s,'On independent data, does removal give a useful smaller model? Check prediction loss and total cost.',650,441,490,112,26);
 line(s,60,577,1080,C.light,1);
 text(s,'Local harm motivates a removal test. Deletion benefit needs separate evidence.',60,595,1080,43,24,C.teal,true);
 note(s,'建议用时 55 秒。第三页直接介绍两个互补验证步骤。第一步，加入检查：用同样的训练配置，比较某个特征组合S以及加入i后的组合，看是否由达标变成不达标。这说明i对该组合有害，不等于它对全部特征或所有情境都天然有害。第二步，删除验证：从当前完整分析输入集合D提出删除候选，在独立数据上检验缩小模型之后的预测损失、压缩程度和总成本，判断整体是否值得。这一步属于计划中的删除决策实验。两步可以概括为先发现问题，再验证行动是否有价值。它们不是随机A与引导B那两个搜索算法，A/B只是第一步寻找上下文的不同方式。整个过程保持监督分类或回归任务不变。',[P+' §1–3 and §7',P+' §22.4']);
}
// Formula explanation before the worked numerical example.
{
 const s=slide('The formulas behind the addition check');
 text(s,'D: full analyzed input set. S: selected inputs without i. i: the input to add.\nU(S): validation score after training on S. Higher is better.',60,157,1080,67,23,C.gray);
 text(s,'q = α U(D)',60,243,490,56,34,C.teal,true);
 text(s,'q is the required score. By default,\nα = 0.95, so q is 95% of U(D).',605,240,535,65,24);
 line(s,60,316,1080,C.light,1);
 text(s,'vα(S) = 1[U(S) ≥ q]',60,331,515,53,31,C.navy,true);
 text(s,'1 means the score meets the requirement.\n0 means it falls below the requirement.',605,325,535,65,24);
 text(s,'Δᵢ(S) = vα(S ∪ {i}) − vα(S)',60,417,520,52,27,C.navy,true);
 text(s,'Measure the change after adding i.\n−1 means a change from pass to fail.',605,405,535,67,24);
 text(s,'Mᵢ = min { Δᵢ(S) }',60,499,520,53,32,C.teal,true);
 text(s,'MinPPI takes the minimum over\nall input groups S without i.',605,491,535,66,24);
 text(s,'Mᵢ = −1 means at least one harmful context exists under the fixed task conditions.',60,594,1080,43,23,C.teal,true);
 note(s,'建议用时 70 秒。按四行讲清公式。先定义D是本次分析的完整输入集合，S是其中一个不包含目标变量i的输入子集，这里的i是待检查的输入特征，不是预测标签y。U(S)表示用S训练模型后的验证集表现，分数越高越好。第一行，完整输入得分U(D)乘上比例alpha得到门槛q，默认alpha是0.95。这里取的是完整输入基准分数的95%，不是要求95%的分类准确率。D仅指本次分析的完整特征范围，例如八维面板，并不必然等于全部原始变量。第二行把连续分数转成是否达标：方括号中的条件成立取1，否则取0。第三行比较加入i前后两个达标状态，变化为负一表示从达标变为不达标。第四行MinPPI在所有不含i的子集上取最小变化，所以只要发现一次负一，精确MinPPI就是负一。找不到负一不等于已证明安全，因为有限采样可能漏掉上下文。所有比较固定数据划分、预处理、学习器、随机性、效用和门槛，要求U(D)>0且分数有限。下一页直接把0.90、0.95、0.87和0.84代入，不再引入新符号。',[P+' §3',paper+' Utility-Threshold Voting Game and Min-Marginal Power Indices']);
}
// 04. The one numerical example used throughout the talk.
{
 const s=slide('A concrete example of the core idea');
 text(s,'Illustration only: α = 0.95, U(D) = 0.90, so q = 0.855',60,164,1080,48,26,C.gray);
 const a=node(s,60,254,420,180,'Model using a subset S','Validation score: 0.87\nMeets the requirement',{accent:true,size:29});
 const b=node(s,720,254,420,180,'Model using S plus input i','Validation score: 0.84\nFalls below the requirement',{size:29});
 connect(s,a,b);
 text(s,'Add input i\nand retrain',500,265,198,95,25,C.navy,true,'center');
 text(s,'Same training rows, learner settings and validation rows',60,461,1080,39,23,C.gray,false,'center');
 claim(s,'Here Δᵢ(S) = 0 − 1 = −1: a harmful witness that another person can check.',527);
 text(s,'It motivates a deletion test. It does not prove that joint deletion will help.',60,606,1080,31,21,C.gray);
 note(s,'建议用时 70 秒。只讲这一个数值例子。完整输入模型假设为0.90，取其95%得到门槛0.855。集合S训练的模型得分0.87，加入i重新训练后为0.84。除了输入集合之外，训练、验证和模型配置保持一致。这个由达标到不达标的变化，就是后面所有方法和实验围绕的证据。数字仅为教学示例，不是实验结果。必须强调，S可能只是完整输入的一部分，不能据此推出从完整集合删除i一定有益。',[P+' §3','Illustrative utilities, not observed experimental results.']);
}
// 05. Central architecture. Two evidence layers with explicit input/output.
{
 const s=slide('Research architecture');
 text(s,'Fixed supervised task: labeled data, learner, split and performance requirement',60,158,1080,47,25,C.navy,true);
 const a=node(s,60,235,285,157,'1. Choose a context','Select input i and\nother inputs S');
 const b=node(s,438,235,325,157,'2. Test the addition','Train on S and on S+i.\nCompare validation scores.',{accent:true,size:23});
 const c=node(s,855,235,285,157,'3. Record failures','Save only pass-to-fail:\ni, S, scores and threshold.');
 connect(s,a,b);connect(s,b,c);
 text(s,'FEATURE AUDIT',60,207,1080,25,17,C.teal,true);
 const d=node(s,855,478,285,115,'4. Propose removal','Choose one candidate.',{size:22});
 const e=node(s,438,478,325,115,'5. Independent check','Accept it or keep all inputs.',{size:22});
 const f=node(s,60,478,285,115,'6. Held-out test','Measure size, loss and cost.',{size:22});
 connect(s,c,d,'bottom','top');connect(s,d,e,'left','right');connect(s,e,f,'left','right');
 text(s,'Repeat the audit within a fixed query budget.',60,404,720,28,21,C.gray);
 text(s,'PLANNED DELETION EVALUATION',60,443,680,27,17,C.teal,true);
 note(s,'建议用时 80 秒。这是整场答辩最重要的一页。按箭头走上排，再从右向左走下排。上排是审计：选一个变量i与上下文S，用两组输入分别训练相同类型的监督模型，检查有没有负向跨越门槛，保存可以复查的记录。有限预算下不断重复前三步。下排是独立的应用验证：把见证转化为一个待删除候选，交给没有参与候选选择的验证集，通过才采用，否则回到完整输入。最后在未用过的测试集上测压缩、预测损失和总成本。两排分别回答“发生过什么”和“是否值得采取行动”。下排属于计划中的决策实验。',[P+' §3–7 and §22.4']);
}
// 06. One theory idea, each symbol immediately explained.
{
 const s=slide('The theory behind MinPPI');
 table(s,[['Game-theoretic idea','Meaning for a prediction model'],['Player','One input column'],['Coalition','A group of columns used to train a model'],['Marginal contribution','The change after adding one column to that group']],[330,750],{y:168,h:240,font:25});
 text(s,'MinPPI checks the worst threshold change across feature groups.',60,438,1080,66,31,C.teal,true);
 text(s,'If any group changes from pass to fail, that input has a negative witness.',60,518,1080,65,27);
 text(s,'Minimum aggregation builds on prior work. Our focus is evidence at a fixed performance requirement.',60,608,1080,29,18,C.gray);
 note(s,'建议用时 60 秒。合作博弈提供一种组织问题的语言：特征是参与者，输入集合是联盟，边际贡献是加入一个变量带来的变化。这里先把分数转成是否达标，再找所有上下文中最差的边际。出现从1到0的变化，最小值就是负一。经典Shapley或Banzhaf关注加权平均边际，本项目关注是否存在一个失败上下文。MinShap已有最小聚合思路，因此不把min本身说成首创。这里对应前面公式页的最小边际定义。',[P+' §2–3','Shapley (1953), A Value for n-Person Games.','Banzhaf (1965), Weighted Voting Doesn’t Work.','Zheng & Raskutti (2026), https://arxiv.org/abs/2604.15107']);
}
// 07. Search has one job: choose contexts cheaply.
{
 const s=slide('Finding witnesses within a limited budget');
 text(s,'Checking every combination becomes expensive as inputs grow.',60,164,1080,62,31,C.navy,true);
 pair(s,60,272,500,'A: random exploration','Try varied groups of inputs.\nThis is the reference search strategy.',{size:27});
 pair(s,650,272,490,'B: guided exploration','Use training associations to propose groups worth testing.',{size:27});
 line(s,60,457,1080,C.light,1);
 text(s,'Both use the same two-model check and the same query budget.',60,481,1080,55,28,C.teal,true);
 text(s,'A found witness is checkable. An unvisited group may still hide one.',60,564,1080,67,27);
 note(s,'建议用时 60 秒。困难不在检查一对模型，而在输入组合太多。A先抽子集大小，再抽该大小内的子集。B使用训练侧互信息和相关结构引导候选上下文。两者只改变“去哪里找”，不会改变监督任务或判定标准。引导是否更有效是实验问题，不能因为使用了相关性就假定更好。找到负见证可以确认固定游戏中的这次失败；没有找到则可能是搜索遗漏。共同查询预算不等于共同实际耗时，因此还要测模型拟合和全流程成本。',[P+' §4',F+' §5']);
}
// 08. Selection rationale precedes inventory.
{
 const s=slide('Why these datasets fit the research');
 text(s,'The core studies need checkable cases, known structure and real prediction tasks.',60,158,1080,56,28,C.navy,true);
 table(s,[['Dataset','Inputs and target','Reason for inclusion'],['Abalone','Physical measurements predict ring count.','Eight inputs allow every subset to be checked. Covers regression.'],['Madelon','Signal, redundant and distractor columns predict a synthetic class.','Known generating structure challenges feature search.'],['Spambase','Word and character statistics predict spam.','Real inputs give interpretable classification examples.'],['Credit-card fraud','PCA components and amount predict fraud.','Rare positive cases test a different classification setting.']],[240,420,420],{y:232,h:323,font:22,rowHeights:[43,70,70,70,70]});
 text(s,'All four are supervised tasks. PCA describes inputs, not a change of learning task.',60,573,1080,36,23,C.teal,true);
 text(s,'The broader benchmark also includes Gisette and ISOLET.',60,620,1080,25,18,C.gray);
 note(s,'建议用时 85 秒。按研究用途解释选择，而不是报六组数字。Abalone只有八个输入，我们可以检查所有组合，且提供回归任务。Madelon是有标签的合成分类数据，包含有信息变量、冗余和干扰，但生成角色不等于MinPPI真值。Spambase的词频和字符统计容易对应到实际输入。Fraud提供稀有正类任务，其28个PCA主成分加金额仍用于有标签监督分类。当前35个精确游戏只来自这四套数据；Gisette和ISOLET属于更广的原始基准范围，并未加入这35个游戏。',[P+' §4',F+' §2–3','https://archive.ics.uci.edu/dataset/1/abalone','https://archive.ics.uci.edu/dataset/171/madelon','https://archive.ics.uci.edu/dataset/94/spambase','https://www.openml.org/api/v1/json/data/1597']);
}
// 09. A map that gives every subsequent experiment a purpose.
{
 const s=slide('Three studies answer three different questions');
 row(s,'01',169,'Discovery: how much do we miss?','Compare the search with all possible subsets in small problems.');
 row(s,'02',312,'Mechanism: what makes discovery difficult?','Control signal, redundancy and interaction in synthetic data.');
 row(s,'03',455,'Decision: is removal worth doing?','Test smaller models on independent data and include the full cost.');
 note(s,'建议用时 45 秒。这三项研究形成证据链，而不是三组互不相关的实验。第一层有精确参照，能判断发现是否充分，已经完成。第二层控制数据结构，解释为什么会漏，处于计划阶段。第三层独立验证删除收益，回答实际使用是否值得，也处于计划阶段。每层的结论都有边界，不能用第一层的结果替代第三层。项目内部对应E1、E2和E4，为答辩按研究问题顺序称Study1、2、3。',[P+' §4, §6, §7, §22',F]);
}
// 10. Full enumeration is a reference, not a method advantage.
{
 const s=slide('Study 1: measure what the search misses');
 text(s,'Small problems let us establish a complete reference.',60,158,1080,60,31,C.navy,true);
 const a=node(s,60,259,310,162,'Same 8-input problem','Same data split, model\nand performance requirement');
 const b=node(s,478,226,322,125,'Complete enumeration','Check all 256 subsets.',{accent:true});
 const c=node(s,478,401,322,125,'Limited-budget search','Let A or B request subsets.');
 const d=node(s,895,310,245,161,'Compare','How many harmful\ninputs were found?');
 connect(s,a,b,'right','left','elbow');connect(s,a,c,'right','left','elbow');connect(s,b,d,'right','left','elbow');connect(s,c,d,'right','left','elbow');
 claim(s,'Success means finding more of the known harmful inputs at the same budget.');
 note(s,'建议用时 60 秒。八个输入共有256个子集，因此能计算完整参照。这是为什么使用小面板：我们需要知道漏了多少，而不是只展示几个找到的案例。A和B不能看到完整答案，只能按预算查询自己选的子集。最后才由评价程序与精确标签比较，主要看整个预算过程的召回。已经完成四数据集、七面板、五划分组成的35个固定游戏。这里的精确答案仅针对各八维面板，不能外推成原始高维任务的真值。没有有害输入时召回没有定义，仍保留这些案例，不挑更容易展示成功的数据。',[P+' §4',F+' §2–4']);
}
// 11. Mechanism, in plain terms before formal data-generating equations.
{
 const s=slide('Study 2: explain why some witnesses are harder to find');
 text(s,'Planned controlled experiment',60,155,1080,40,22,C.gray);
 table(s,[['Controlled structure','What changes in the inputs','What we learn'],['Signal with distractors','Useful variables plus unrelated columns.','Whether extra noise creates difficult contexts.'],['Correlated redundancy','Add noisy copies of informative variables.','How overlapping information affects discovery.'],['Interaction','The target depends on a combination of variables.','Whether the search reaches important combinations.']],[290,425,365],{y:224,h:297,font:23,rowHeights:[48,83,83,83]});
 claim(s,'Keep the model and evaluation fixed. Change the data structure deliberately.');
 note(s,'建议用时 55 秒。真实数据看到差异后很难解释原因，所以第二项研究有目的地构造三个监督二分类机制。第一个是信号加无关变量，第二个增加带噪声的相关副本，第三个是两变量异或交互。仍然做完整枚举作为参照，再比较同预算A/B。每种五个预先固定实例，不根据结果换随机种子。这里“无关”或“冗余”是生成角色，不能直接当成有害标签。研究要检验这些结构如何改变见证和搜索难度，而不是预先假定任何一种一定有害。',[P+' §22.3–23','../../../../Master-mr2/MR3_E2_PLAN.md']);
}
// 12. Practical value requires genuinely separate evaluation data.
{
 const s=slide('Study 3: test whether removal is useful');
 text(s,'Planned decision experiment with separate data for each decision stage',60,157,1080,55,26,C.gray);
 const a=node(s,60,270,305,155,'Choose a candidate','Use training and\nselection-validation data.');
 const b=node(s,447,270,305,155,'Check independently','Accept within tolerance.\nOtherwise keep all inputs.',{accent:true,size:22});
 const c=node(s,835,270,305,155,'Freeze and test','Refit the chosen model.\nEvaluate on held-out data.');
 connect(s,a,b);connect(s,b,c);
 text(s,'Compare with established feature-ranking and elimination methods.',60,468,1080,48,26,C.navy,true);
 claim(s,'Judge three outcomes together: fewer inputs, prediction loss and total cost.');
 note(s,'建议用时 80 秒。第三项研究最接近实际使用价值。每个策略在选择验证集上只提交一个候选，再由独立gate判断是否满足事先设定的预测损失容忍度。通过采用候选，不通过回完整输入，不能反复试到通过。冻结决定后在非测试数据上重训，再测试。比较包括互信息、TreeSHAP、LOFO、RFE和由验证集选择这些候选的组合策略，主比较是MinPPI-A与该组合策略。公平性要求同一数据集内固定目标、划分、学习器、预处理规则、指标和容忍度，并纳入所有成本。分类AUC和回归R²分开解释，不能把其原始分数混合平均。',[P+' §7 and §22.4']);
}
// 13. Current evidence is one honest page.
{
 const s=slide('Current evidence and what it means');
 row(s,'01',170,'The reference audit is complete','Small problems provide checkable witnesses and reveal missed cases.');
 row(s,'02',313,'Guidance has no stable advantage yet','Training associations do not consistently point toward failure contexts.');
 row(s,'03',456,'Practical deletion value is still open','Controlled mechanisms and independent removal tests are the next evidence.');
 note(s,'建议用时 45 秒。结果只讲三个认识。第一，35个小型游戏已完成，能够检查见证和遗漏。第二，当前这种B引导没有稳定优于随机A，不能包装成已证实的加速或优势。第三，是否值得删除还没有独立决策证据，因此接下来先做机制实验，再做独立删除实验。这里的负结果让我们更准确地限定算法适用条件，不推翻固定游戏中的见证定义。证据状态参考2026年9月26日台账。',[F+' §1–8',W]);
}
// 14. End with the value of the research, not a parameter list.
{
 const s=slide('Expected contribution',{dark:true});
 text(s,'A feature-selection decision\nwith evidence that can be inspected.',60,170,1080,124,45,C.white,true);
 text(s,'A concrete failure context explains what happened.\nAn exact reference shows what the search missed.\nAn independent test shows whether removal is worthwhile.',60,343,1080,180,29,C.mint);
 line(s,60,553,1080,'#496174',1);
 text(s,'Next: controlled mechanisms, then independent deletion evaluation.',60,578,1080,53,25,C.white);
 note(s,'建议用时 45 秒。结尾回到价值。希望交付的不只是一个重要性分数，而是一组可以核查的理由：哪个输入在什么组合下造成达标失败；有限预算搜索遗漏多少；采取删除行动是否值得其成本。前两者属于方法与发现评价，最后一项需要下一步实验支持。',[P+' §1, §7 and §22',F]);
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
 workspaceDir,candidatePath,finalPath:path.join(workspaceDir,'revised','MinPPI_Defense_Wenjie_SHI_Main_15_Slides.pptx'),
 pythonExecutable:RUNTIME_PYTHON,
 integrityValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','15240000,8572500','--validate-bullet-geometry','--validate-heading-fit',...nativeTableSlides.flatMap(n=>['--require-native-table-slide',String(n)])],
 explicitTotalSlideCount:15,
 requiredNativeTableOwnerSlides:nativeTableSlides,
 requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'design',families:['Arial'],scriptFonts:{ea:'Microsoft YaHei'}},
 verifyArtifactToolImport:true,
 receiptPath:path.join(tmp,'validation-main-only.json')
});
console.log(JSON.stringify(result));
const finalDeck=await PresentationFile.importPptx(await FileBlob.load(result.finalPath));
for(let i=0;i<finalDeck.slides.items.length;i++){
 const png=await finalDeck.export({slide:finalDeck.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(tmp,`final-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
console.log('FINAL_RENDERED',finalDeck.slides.items.length);
