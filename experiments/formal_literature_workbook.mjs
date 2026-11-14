/** Build the formal study workbook from the audited, typed report snapshot. */
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';
import fs from 'node:fs/promises';
import path from 'node:path';

const [input,out] = process.argv.slice(2);
const {rows,groups,features} = JSON.parse(await fs.readFile(input,'utf8'));
const book=Workbook.create();
const order=['abalone','spambase','madelon','credit_card_fraud','gisette','isolet'];
const datasets={abalone:'Abalone',spambase:'Spambase',madelon:'Madelon',credit_card_fraud:'Credit Card Fraud',gisette:'Gisette',isolet:'ISOLET'};
const modes={shared:'共享参照',native:'文献估计量',frozen:'统一冻结效用'};
const methods={full:'全列模型',minpi:'minPI',gain:'紧凑 Gain',pfi:'快速 PFI',minshap:'MinShap',shapley:'排列 Shapley',sampling_shap:'SamplingSHAP',banzhaf:'采样 Banzhaf',shapg:'ShapG'};
const status={completed:'完成',in_progress:'进行中',not_started:'未启动',failed:'失败',timed_out:'超时',interrupted:'中断'};
const groupOrder=(a,b)=>order.indexOf(a.dataset)-order.indexOf(b.dataset)
    ||Object.keys(modes).indexOf(a.mode)-Object.keys(modes).indexOf(b.mode)
    ||Object.keys(methods).indexOf(a.method)-Object.keys(methods).indexOf(b.method);
rows.sort((a,b)=>groupOrder(a,b)||a.seed-b.seed);
groups.sort(groupOrder);
features.sort((a,b)=>groupOrder(a,b)||a.seed-b.seed||a.selected_position-b.selected_position);
const count=rows.filter(row=>row.status==='completed').length;
const column=n=>{
  let label='';
  for(;n;n=Math.floor((n-1)/26))label=String.fromCharCode(65+(n-1)%26)+label;
  return label;
};
const layouts=[];
function sheet(name,id,title,note,headers,values,accent){
    const tab=book.worksheets.add(name);
    tab.showGridLines=false;tab.tabColor=accent;
    const last=column(headers.length),end=4+values.length;
    const all=tab.getRange('A1:'+last+Math.max(5,end));
    all.format.font={name:'Microsoft YaHei',size:10,color:'#21353E'};
    all.format.rowHeightPx=24;all.format.verticalAlignment='center';
    for(const [row,description,height,size] of [[1,title,38,18],[2,note,28,10]]){
        const cell=tab.getRange('A'+row+':'+column(Math.min(headers.length,10))+row);
        cell.merge();cell.values=[[description]];cell.format.rowHeightPx=height;
        cell.format.font={name:'Microsoft YaHei',size,color:accent,bold:row===1};
    }
    tab.getRange('A3').format.rowHeightPx=8;
    tab.getRange('A4:'+last+end).values=[headers.map(h=>h[0]),...values];
    if(values.length){
        const table=tab.tables.add('A4:'+last+end,true,id);
        table.style='TableStyleLight9';table.showFilterButton=true;
    }
    const top=tab.getRange('A4:'+last+'4');
    top.format={fill:accent,font:{name:'Microsoft YaHei',size:10,bold:true,color:'#FFFFFF'},verticalAlignment:'center',horizontalAlignment:'center'};
    top.format.rowHeightPx=34;
    for(const [idx,[,width,format]] of headers.entries()){
        const c=column(idx+1);tab.getRange(c+'1').format.columnWidthPx=width;
        if(format&&values.length)tab.getRange(c+'5:'+c+end).setNumberFormat(format);
    }
    tab.freezePanes.freezeRows(4);tab.freezePanes.freezeColumns(2);
    layouts.push({name,rows:values,headers:headers.map(h=>h[0]),end});
    return tab;
}
const integer='#,##0',decimal='0.00',score='0.0000',percent='0.00%';
const predictions = 'sample_prediction_rows' in rows[0] ? [
    ['sample_prediction_rows','采样预测行','V',decimal],
    ['sample_prediction_calls','采样调用','W',decimal],
    ['selection_prediction_rows','选择总预测行','X',decimal],
    ['selection_prediction_calls','选择总调用','Y',decimal],
] : [];
const summaryMeasures=[
    ['k','保留列','G',decimal],['deleted','删除列','H',decimal],['fits','拟合次数','I',decimal],
    ['wall_s','总秒','J',decimal],['auc','AUC','K',score],['accuracy','准确率','L',percent],
    ['ap','AP','M',score],['r2','R²','N',score],['mae','MAE','O',score], ...predictions,
];
const sumHeaders=[['数据集',153],['求值方式',150],['方法',161],['已完成',105,integer],['计划',94,integer],['输入列范围',125],
    ...summaryMeasures.flatMap(([,label,,fmt])=>[[label+'均值',124,fmt],[label+'标准差',135,fmt]])];
const sumRows=groups.map(g=>[datasets[g.dataset],modes[g.mode],methods[g.method],g.n_completed,g.n_planned,
    g.width_min===g.width_max?g.width_min:g.width_min+'–'+g.width_max,
    ...summaryMeasures.flatMap(([f])=>[g[f+'_mean']??null,g[f+'_sd']??null])]);
const seeds=[...new Set(rows.map(r=>r.seed))];
const seedLabel=seeds.length===1?'seed '+seeds[0]:seeds.length+' 个 seed';
const summary=sheet('跨 seed 汇总','AcrossSeeds','预处理数据 · '+seedLabel+' 的正式实验',
    `已完成 ${count} / ${rows.length} 条独立路线。样本标准差使用 n−1。总秒包含冷启动与前置筛选，回归 R² 使用未截断值。`,
    sumHeaders,sumRows,'#26495A');
const detailHeaders=[['数据集',152],['求值方式',148],['方法',156],['Seed',91,integer],['状态',100],['输入列',110,integer],['保留列',110,integer],
    ['删除列',110,integer],['拟合次数',124,integer],['冷启动总秒',141,decimal],['测试 AUC',114,score],['测试准确率',130,percent],
    ['测试 AP',107,score],['测试 R²',105,score],['测试 MAE',120,score],['全列 AUC',117,score],['全列准确率',133,percent],
    ['全列 AP',110,score],['全列 R²',106,score],['全列 MAE',121,score],['选择用时秒',134,decimal],
    ...predictions.map(([,label])=>[label,140,integer]),...(predictions.length?[['采样配置',460]]:[])];
const testKeys=['auc','accuracy','ap','r2','mae'];
const detailKeys=[...testKeys,...testKeys.map(k=>'full_'+k),'selector_s',...predictions.map(p=>p[0])];
if(predictions.length)detailKeys.push('sampling_policy');
const details=rows.map(r=>[datasets[r.dataset],modes[r.mode],methods[r.method],r.seed,status[r.status]||r.status,r.width,r.k??null,
    r.deleted??null,r.fits??null,r.wall_s??null,...detailKeys.map(k=>r[k]??null)]);
const detail=sheet('逐 seed 明细','BySeed','每次独立运行的列数、成本与测试得分',
    '全列成绩对应同一 seed 和数据集。欺诈检测另列 AP，ISOLET 使用宏平均 OvR AUC。来源为各路线的预测文件、拟合记录和独立冷启动计时。',
    detailHeaders,details,'#32645C');
const key=r=>[r.dataset,r.mode,r.method].join('/');
const extents=new Map();
for(const [i,r] of rows.entries()){
    const k=key(r),cell=i+5;
    if(!extents.has(k))extents.set(k,[cell,cell]);
    else extents.get(k)[1]=cell;
    if(r.status==='completed')detail.getRange('H'+cell).formulas=[['=F'+cell+'-G'+cell]];
}
const tabRef="'逐 seed 明细'!";
for(const [i,r] of groups.entries()){
    const [first,last]=extents.get(key(r));
    for(const [j,[measure,,c]] of summaryMeasures.entries()){
        const cells=tabRef+c+first+':'+c+last;
        if(r[measure+'_mean']==null)continue;
        summary.getRange(column(7+j*2)+(i+5)).formulas=[['=AVERAGE('+cells+')']];
        if(r[measure+'_sd']!=null)summary.getRange(column(8+j*2)+(i+5)).formulas=[['=STDEV('+cells+')']];
    }
}
const featHeaders=[['数据集',151],['求值方式',147],['方法',147],['Seed',104,integer],['子集内序号',136,integer],['预处理列序号',154,integer],['特征名',335]];
sheet('保留特征','SelectedFeatures','每个已完成模型的输入列',
    '可按数据集、方法、求值方式和 seed 筛选。两种序号均从 1 开始。完整子集为空的模型在明细中显示保留 0 列。',
    featHeaders,features.map(r=>[datasets[r.dataset],modes[r.mode],methods[r.method],r.seed,r.selected_position,r.source_position,r.feature]),'#70543E');
book.recalculate();
for(const layout of layouts){
    if(!layout.rows.length)continue;
    const tab=book.worksheets.getItem(layout.name);
    const values=tab.getRange('A5:'+column(layout.headers.length)+layout.end).values;
    layout.rows.forEach((row,i)=>row.forEach((v,j)=>{
        const found=values[i][j]??null;
        if(typeof v==='number'?!(typeof found==='number'&&Math.abs(v-found)<1e-8):found!==v){
            throw Error(layout.name+' '+column(j+1)+(i+5)+': '+String(found)+' vs '+String(v));
        }
    }));
    const picture=await book.render({sheetName:layout.name,range:'A1:'+column(Math.min(layout.headers.length,10))+Math.min(layout.end,13),scale:.85,format:'png'});
    await fs.writeFile(path.join(path.dirname(input),layout.name+'.png'),new Uint8Array(await picture.arrayBuffer()));
}
const exported=await SpreadsheetFile.exportXlsx(book);
await exported.save(out);
