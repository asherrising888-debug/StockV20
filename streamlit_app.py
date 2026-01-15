import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta

# ==================== ⚙️ 核心配置 (V20.2 Web版) ====================
st.set_page_config(page_title="V20.2 战略指挥舱", layout="wide", page_icon="🚀")

STOCK_DIR = "./stock_data_v20"

# 2030 战略核心资产池
STRATEGIC_POOL = {
    "002230": ("科大讯飞", "AI模型"), "688256": ("寒武纪", "AI芯片"),
    "000977": ("浪潮信息", "服务器"), "603019": ("中科曙光", "超算"),
    "601138": ("工业富联", "AI服务器"), "600588": ("用友网络", "软件"),
    "688111": ("金山办公", "办公AI"),   "600570": ("恒生电子", "金融IT"),
    "300308": ("中际旭创", "CPO"),     "000063": ("中兴通讯", "6G"),
    "688027": ("国盾量子", "量子"),     "000066": ("中国长城", "信创"),
    "600050": ("中国联通", "数据"),     "601728": ("中国电信", "数据"),
    "600941": ("中国移动", "数据"),
    "688981": ("中芯国际", "晶圆"),     "002371": ("北方华创", "设备"),
    "603501": ("韦尔股份", "设计"),     "002049": ("紫光国微", "军工芯"),
    "688126": ("沪硅产业", "材料"),     "603986": ("兆易创新", "存储"),
    "600118": ("中国卫星", "航天"),     "600893": ("航发动力", "发动机"),
    "002085": ("万丰奥威", "低空"),     "600038": ("中直股份", "直升机"),
    "000099": ("中信海直", "低空"),     "688070": ("纵横股份", "无人机"),
    "002625": ("光启技术", "超材料"),   "600343": ("航天动力", "航天"),
    "600760": ("中航沈飞", "军工"),     "002179": ("中航光电", "连接器"),
    "600150": ("中国船舶", "造船"),
    "300124": ("汇川技术", "工控"),     "002747": ("埃斯顿", "机器人"),
    "601882": ("海天精工", "机床"),     "600031": ("三一重工", "机械"),
    "002475": ("立讯精密", "果链"),     "000725": ("京东方A", "面板"),
    "000100": ("TCL科技", "面板"),      "002050": ("三花智控", "汽零"),
    "300750": ("宁德时代", "电池"),     "002594": ("比亚迪", "汽车"),
    "300014": ("亿纬锂能", "电池"),     "300274": ("阳光电源", "储能"),
    "601012": ("隆基绿能", "光伏"),     "600438": ("通威股份", "光伏"),
    "002202": ("金风科技", "风电"),     "688339": ("亿华通", "氢能"),
    "600900": ("长江电力", "水电"),     "601985": ("中国核电", "核电"),
    "600027": ("华电国际", "火电"),     "600989": ("宝丰能源", "氢能"),
    "600276": ("恒瑞医药", "创新药"),   "603259": ("药明康德", "CXO"),
    "688065": ("凯赛生物", "合成生物"), "688363": ("华熙生物", "医美"),
    "300760": ("迈瑞医疗", "器械"),     "300676": ("华大基因", "基因"),
    "688315": ("诺禾致源", "测序"),     "000538": ("云南白药", "中药"),
    "600519": ("贵州茅台", "白酒"),     "000858": ("五粮液", "白酒"),
    "601888": ("中国中免", "免税"),     "601919": ("中远海控", "航运"),
    "601899": ("紫金矿业", "有色"),     "600030": ("中信证券", "金融"),
    "000333": ("美的集团", "家电"),     "601668": ("中国建筑", "基建")
}
for k, v in STRATEGIC_POOL.items():
    if isinstance(v, str): STRATEGIC_POOL[k] = (v, "其他")

PARAMS = {
    'MA_LIFE': 20, 'MA_BULL': 40, 'RSI_N': 14, 'ATR_N': 14, 'VOL_MA': 20,
    'BIAS_LIMIT': 1.12, 'RSI_MIN': 50, 'RSI_MAX': 75,
    'VOL_MIN': 1.0, 'VOL_MAX': 2.5, 'STOP_LOSS': -0.08
}

# ==================== 核心算法 ====================
class AlgoEngine:
    @staticmethod
    def get_snapshot():
        try:
            df = ak.stock_zh_a_spot_em()
            snap = {}
            for _, row in df.iterrows():
                snap[str(row['代码'])] = {
                    'close': float(row['最新价']), 'high': float(row['最高']),
                    'low': float(row['最低']), 'open': float(row['今开']),
                    'volume': float(row['成交量'])
                }
            return snap
        except: return None

    @staticmethod
    def sync_history():
        if not os.path.exists(STOCK_DIR): os.makedirs(STOCK_DIR)
        
        # 使用 Streamlit 的 status 组件显示进度
        status = st.status("📡 正在同步数据...", expanded=True)
        
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=800)).strftime("%Y%m%d")
        
        # 1. 大盘
        try:
            status.write("正在下载沪深300指数...")
            try:
                df = ak.stock_zh_index_daily_em(symbol="sh000300")
            except:
                df = ak.stock_zh_index_daily(symbol="sh000300")
            
            rename_map = {'date': '日期', 'close': '收盘', 'open': '开盘', 'high': '最高', 'low': '最低', 'volume': '成交量'}
            df.rename(columns=rename_map, inplace=True)
            df.to_csv(os.path.join(STOCK_DIR, "sh000300.csv"), index=False)
        except Exception as e:
            status.write(f"⚠️ 大盘同步警告: {e}")

        # 2. 个股
        status.write(f"正在同步 {len(STRATEGIC_POOL)} 只核心资产...")
        progress_bar = status.progress(0)
        
        cnt = 0
        total = len(STRATEGIC_POOL)
        for i, code in enumerate(STRATEGIC_POOL.keys()):
            try:
                df = ak.stock_zh_a_hist(symbol=code, start_date=start, end_date=end, adjust="qfq")
                if not df.empty:
                    df.to_csv(os.path.join(STOCK_DIR, f"{code}.csv"), index=False)
                    cnt += 1
            except: pass
            progress_bar.progress((i + 1) / total)
            
        status.update(label=f"✅ 同步完成！覆盖 {cnt} 只股票。", state="complete", expanded=False)

    @staticmethod
    def get_market_status():
        path = os.path.join(STOCK_DIR, "sh000300.csv")
        if not os.path.exists(path): return False, 0, 0, "无数据，请先同步"
        try:
            df = pd.read_csv(path)
            if 'date' in df.columns: df.rename(columns={'date':'日期', 'close':'收盘'}, inplace=True)
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            
            df_w = df.resample('W-FRI').agg({'收盘': 'last'})
            df_w['MA40'] = df_w['收盘'].rolling(PARAMS['MA_BULL']).mean()
            
            last = df_w.iloc[-1]
            prev = df_w.iloc[-2]
            
            is_bull = (last['收盘'] > last['MA40']) and (last['MA40'] >= prev['MA40'] * 0.9995)
            date_str = df_w.index[-1].strftime("%Y-%m-%d")
            return is_bull, last['收盘'], last['MA40'], date_str
        except Exception as e: return False, 0, 0, str(e)

    @staticmethod
    def calc_indicators(code, snapshot):
        path = os.path.join(STOCK_DIR, f"{code}.csv")
        if not os.path.exists(path): return None
        try:
            df = pd.read_csv(path)
            rename_map = {'日期':'date', '开盘':'open', '收盘':'close', '最高':'high', '最低':'low', '成交量':'volume'}
            df.rename(columns=rename_map, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            if snapshot and code in snapshot:
                real = snapshot[code]
                today = datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
                if today not in df.index:
                    df.loc[today] = [real['open'], real['close'], real['high'], real['low'], real['volume']] + [0]*(len(df.columns)-5)
                else:
                    df.loc[today, ['open','close','high','low','volume']] = [real['open'], real['close'], real['high'], real['low'], real['volume']]
            
            df_w = df.resample('W-FRI').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
            close = df_w['close']
            
            df_w['MA20'] = close.rolling(PARAMS['MA_LIFE']).mean()
            df_w['MA20_Up'] = df_w['MA20'] > df_w['MA20'].shift(1)
            df_w['Vol_MA20'] = df_w['volume'].rolling(PARAMS['VOL_MA']).mean()
            
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).ewm(com=PARAMS['RSI_N']-1, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(com=PARAMS['RSI_N']-1, adjust=False).mean()
            rs = gain / loss
            df_w['RSI'] = 100 - (100 / (1 + rs))
            
            tr1 = df_w['high'] - df_w['low']
            tr2 = abs(df_w['high'] - close.shift(1))
            tr3 = abs(df_w['low'] - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df_w['ATR'] = tr.ewm(com=PARAMS['ATR_N']-1, adjust=False).mean()
            
            cur = df_w.iloc[-1].to_dict()
            cur['code'] = code
            cur['name'] = STRATEGIC_POOL[code][0]
            cur['ind'] = STRATEGIC_POOL[code][1]
            cur['date_str'] = df_w.index[-1].strftime("%Y-%m-%d")
            
            cur['Bias'] = cur['close'] / cur['MA20'] if cur['MA20'] else 0
            cur['Vol_Ratio'] = cur['volume'] / cur['Vol_MA20'] if cur['Vol_MA20']>0 else 0
            cur['Amount'] = cur['close'] * cur['volume']
            
            body = abs(cur['close'] - cur['open'])
            upper = cur['high'] - max(cur['open'], cur['close'])
            cur['Structure_OK'] = body >= upper
            
            return cur
        except: return None

# ==================== Web 界面 ====================
st.title("🚀 A股 V20.2 实战指挥舱")
st.markdown("---")

# 侧边栏配置
with st.sidebar:
    st.header("1. 战前整备")
    if st.button("🔄 同步最新数据 (周五必点)", type="primary"):
        # === 修复点：调用时不传参数 ===
        AlgoEngine.sync_history()
        
    cash = st.number_input("可用资金 (元):", value=20000.0, step=1000.0)
    mode = st.radio("策略模式:", ["V12 激进 (梭哈)", "V11 稳健 (半仓)"])
    
    st.markdown("### 持仓录入")
    st.caption("格式: 代码,成本,股数,最高价 (V12必填)")
    pos_input = st.text_area("输入:", height=100, placeholder="601138, 22.5, 500, 25.0")

# 主程序
if st.button("🚀 启动全流程诊断", use_container_width=True):
    
    # 0. 解析持仓
    positions = []
    if pos_input:
        for line in pos_input.split('\n'):
            p = line.replace('，', ',').split(',')
            if len(p)>=3:
                try: positions.append({'code':p[0].strip(), 'cost':float(p[1]), 'shares':int(p[2]), 'high':float(p[3]) if len(p)>3 else float(p[1])})
                except: pass

    # 获取快照
    snapshot = AlgoEngine.get_snapshot()
    if not snapshot: st.error("⚠️ 实时数据获取失败，使用历史数据近似。")
    
    # --- Step 1: 环境 ---
    st.subheader("📊 Step 1: 市场环境")
    is_bull, idx_price, idx_ma, idx_date = AlgoEngine.get_market_status()
    
    if idx_price == 0:
        st.error(f"❌ 数据错误: {idx_date}。请先点击左侧【同步最新数据】！")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("沪深300", f"{idx_price:.2f}")
        col2.metric("牛熊线 (MA40)", f"{idx_ma:.2f}")
        col3.metric("状态", "🟢 牛市" if is_bull else "🔴 熊市")
        st.caption(f"数据基准日: {idx_date}")

    # --- Step 2: 持仓 ---
    st.subheader("🛡️ Step 2: 持仓诊断")
    simulated_cash = cash
    active_pos = 0
    
    if positions:
        for p in positions:
            d = AlgoEngine.calc_indicators(p['code'], snapshot)
            if not d: continue
            
            price = d['close']
            pct = (price - p['cost']) / p['cost'] if p['cost']!=0 else 0
            
            reason = None
            if pct <= PARAMS['STOP_LOSS']: reason = f"硬止损(亏{pct:.1%})"
            elif price < d['MA20'] and not d['MA20_Up']: reason = "趋势破坏"
            
            if "V12" in mode:
                stop_line = p['high'] - (3.0 * d['ATR'])
                if price < stop_line: reason = f"ATR止盈(破{stop_line:.2f})"
            
            c1, c2 = st.columns([3, 1])
            with c1:
                st.write(f"**{d['name']}** ({p['code']})")
                st.caption(f"现价:{price} | 成本:{p['cost']} | 盈亏:{pct:.2%}")
                if "V12" in mode: st.caption(f"最高价:{p['high']} | 止盈线:{stop_line:.2f}")
            with c2:
                if reason:
                    st.error(f"❌ 卖出\n{reason}")
                    simulated_cash += price * p['shares']
                else:
                    st.success("✅ 持有")
                    active_pos += 1
                    if price > p['high']: st.info("创新高!请更新")
            st.divider()
    else:
        st.info("当前空仓")

    # --- Step 3: 选股 ---
    if is_bull:
        st.subheader("🔍 Step 3: 选股全景透视")
        
        candidates = []
        table_data = []
        
        # 进度条
        progress_text = "正在扫描 60+ 只核心资产..."
        my_bar = st.progress(0, text=progress_text)
        total_scan = len(STRATEGIC_POOL)
        
        for i, code in enumerate(STRATEGIC_POOL):
            my_bar.progress((i + 1) / total_scan)
            if any(p['code'] == code for p in positions): continue
            
            d = AlgoEngine.calc_indicators(code, snapshot)
            if not d: continue
            
            res = "❌"
            why = []
            
            if not (d['MA20_Up'] and d['close'] > d['MA20']): why.append("MA20向下")
            if d['Bias'] > PARAMS['BIAS_LIMIT']: why.append(f"位置高({d['Bias']:.2f})")
            if not (PARAMS['RSI_MIN'] <= d['RSI'] <= PARAMS['RSI_MAX']): why.append(f"RSI({d['RSI']:.0f})")
            if not (PARAMS['VOL_MIN'] <= d['Vol_Ratio'] <= PARAMS['VOL_MAX']): why.append(f"量({d['Vol_Ratio']:.1f})")
            if not d['Structure_OK']: why.append("结构差")
            if d['close']*100 > simulated_cash: why.append("买不起")
            
            if not why:
                res = "✅"
                candidates.append(d)
                
            table_data.append({
                "代码": code, "名称": d['name'], "现价": f"{d['close']:.2f}",
                "RSI": f"{d['RSI']:.1f}", "MA20": "⬆️" if d['MA20_Up'] else "⬇️",
                "诊断": res, "原因": " ".join(why)
            })
            
        my_bar.empty()
        
        # 显示透视表
        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True, hide_index=True)

        # --- Step 4: 决策 ---
        st.subheader("💡 Step 4: 最终指令")
        
        if not candidates:
            st.warning("扫描结束，无符合V19标准标的。")
        else:
            candidates.sort(key=lambda x: (x['RSI'], x['Amount']), reverse=True)
            target = candidates[0]
            
            invest = simulated_cash * 0.5 if "V11" in mode or active_pos == 0 else simulated_cash * 0.99
            if active_pos >= 2 and "V12" in mode:
                st.warning("V12仓位已满，停止买入。")
            else:
                shares = int(invest / target['close'] / 100) * 100
                if shares >= 100:
                    st.success(f"⭐⭐⭐ 买入指令: {target['name']} ({target['code']})")
                    st.write(f"数量: **{shares}** 股 | RSI: **{target['RSI']:.1f}**")
                    st.caption(f"预计耗资: {shares * target['close']:.2f} 元")
                else:
                    st.error(f"选中 {target['name']}，但资金不足买入一手。")
    else:
        st.error("大盘红灯，停止选股。")
