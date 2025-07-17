import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch
import plotly.graph_objects as go
import plotly.express as px

# ページ設定
st.set_page_config(
    page_title="線形代数デモアプリ",
    page_icon="📐",
    layout="wide"
)

# タイトル
st.title("📐 線形代数デモアプリ")
st.markdown("### 東京理科大学 線形代数X 2025")

# サイドバーでトピック選択
topic = st.sidebar.selectbox(
    "トピックを選択してください",
    ["ベクトルの演算", "行列の演算", "固有値と固有ベクトル", "線形変換"]
)

# セッション状態の初期化
if 'computed' not in st.session_state:
    st.session_state.computed = False

if topic == "ベクトルの演算":
    st.header("ベクトルの演算と可視化")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ベクトル入力")
        # 2次元ベクトル
        st.write("ベクトル a:")
        a1 = st.number_input("a₁", value=3.0, key="a1")
        a2 = st.number_input("a₂", value=2.0, key="a2")
        
        st.write("ベクトル b:")
        b1 = st.number_input("b₁", value=1.0, key="b1")
        b2 = st.number_input("b₂", value=4.0, key="b2")
    
    with col2:
        st.subheader("演算結果")
        
        # ベクトル定義
        a = np.array([a1, a2])
        b = np.array([b1, b2])
        
        # 演算
        st.write(f"**a + b** = [{a[0] + b[0]:.2f}, {a[1] + b[1]:.2f}]")
        st.write(f"**a - b** = [{a[0] - b[0]:.2f}, {a[1] - b[1]:.2f}]")
        st.write(f"**a · b** (内積) = {np.dot(a, b):.2f}")
        st.write(f"**|a|** (ノルム) = {np.linalg.norm(a):.2f}")
        st.write(f"**|b|** (ノルム) = {np.linalg.norm(b):.2f}")
    
    # ベクトルの可視化
    st.subheader("ベクトルの可視化")
    
    fig = go.Figure()
    
    # ベクトルa
    fig.add_trace(go.Scatter(
        x=[0, a[0]], y=[0, a[1]],
        mode='lines+markers',
        name='ベクトル a',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    
    # ベクトルb
    fig.add_trace(go.Scatter(
        x=[0, b[0]], y=[0, b[1]],
        mode='lines+markers',
        name='ベクトル b',
        line=dict(color='red', width=3),
        marker=dict(size=8)
    ))
    
    # ベクトルa+b
    fig.add_trace(go.Scatter(
        x=[0, a[0]+b[0]], y=[0, a[1]+b[1]],
        mode='lines+markers',
        name='a + b',
        line=dict(color='green', width=3, dash='dash'),
        marker=dict(size=8)
    ))
    
    # グリッド設定
    max_val = max(abs(a[0]), abs(a[1]), abs(b[0]), abs(b[1]), 
                  abs(a[0]+b[0]), abs(a[1]+b[1])) + 2
    
    fig.update_layout(
        xaxis=dict(range=[-max_val, max_val], zeroline=True, zerolinewidth=2),
        yaxis=dict(range=[-max_val, max_val], zeroline=True, zerolinewidth=2),
        width=600, height=600,
        showlegend=True
    )
    
    st.plotly_chart(fig)

elif topic == "行列の演算":
    st.header("行列の演算")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("行列A (2×2)")
        a11 = st.number_input("A₁₁", value=2.0, key="a11")
        a12 = st.number_input("A₁₂", value=1.0, key="a12")
        a21 = st.number_input("A₂₁", value=3.0, key="a21")
        a22 = st.number_input("A₂₂", value=4.0, key="a22")
        
        A = np.array([[a11, a12], [a21, a22]])
        st.write("行列A:")
        st.write(pd.DataFrame(A, columns=['列1', '列2'], index=['行1', '行2']))
    
    with col2:
        st.subheader("行列B (2×2)")
        b11 = st.number_input("B₁₁", value=1.0, key="b11")
        b12 = st.number_input("B₁₂", value=2.0, key="b12")
        b21 = st.number_input("B₂₁", value=0.0, key="b21")
        b22 = st.number_input("B₂₂", value=1.0, key="b22")
        
        B = np.array([[b11, b12], [b21, b22]])
        st.write("行列B:")
        st.write(pd.DataFrame(B, columns=['列1', '列2'], index=['行1', '行2']))
    
    st.subheader("演算結果")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("**A + B:**")
        st.write(pd.DataFrame(A + B, columns=['列1', '列2'], index=['行1', '行2']))
        
        st.write("**A × B:**")
        st.write(pd.DataFrame(A @ B, columns=['列1', '列2'], index=['行1', '行2']))
    
    with col4:
        det_A = np.linalg.det(A)
        st.write(f"**det(A)** = {det_A:.3f}")
        
        if abs(det_A) > 1e-10:
            st.write("**A⁻¹ (逆行列):**")
            A_inv = np.linalg.inv(A)
            st.write(pd.DataFrame(A_inv, columns=['列1', '列2'], index=['行1', '行2']))
        else:
            st.write("Aは特異行列です（逆行列は存在しません）")

elif topic == "固有値と固有ベクトル":
    st.header("固有値と固有ベクトル")
    
    st.subheader("行列入力 (2×2)")
    col1, col2 = st.columns(2)
    
    with col1:
        m11 = st.number_input("M₁₁", value=3.0, key="m11")
        m12 = st.number_input("M₁₂", value=1.0, key="m12")
    
    with col2:
        m21 = st.number_input("M₂₁", value=1.0, key="m21")
        m22 = st.number_input("M₂₂", value=3.0, key="m22")
    
    M = np.array([[m11, m12], [m21, m22]])
    
    st.write("行列M:")
    st.write(pd.DataFrame(M, columns=['列1', '列2'], index=['行1', '行2']))
    
    # 固有値と固有ベクトルの計算
    eigenvalues, eigenvectors = np.linalg.eig(M)
    
    st.subheader("計算結果")
    st.write(f"**固有値**: λ₁ = {eigenvalues[0]:.3f}, λ₂ = {eigenvalues[1]:.3f}")
    
    col3, col4 = st.columns(2)
    with col3:
        st.write("**固有ベクトル v₁** (λ₁に対応):")
        st.write(f"[{eigenvectors[0, 0]:.3f}, {eigenvectors[1, 0]:.3f}]")
    
    with col4:
        st.write("**固有ベクトル v₂** (λ₂に対応):")
        st.write(f"[{eigenvectors[0, 1]:.3f}, {eigenvectors[1, 1]:.3f}]")
    
    # 固有ベクトルの可視化
    st.subheader("固有ベクトルの可視化")
    
    fig = go.Figure()
    
    # 固有ベクトル1
    v1 = eigenvectors[:, 0]
    fig.add_trace(go.Scatter(
        x=[0, v1[0]], y=[0, v1[1]],
        mode='lines+markers',
        name=f'v₁ (λ={eigenvalues[0]:.2f})',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    
    # 固有ベクトル2
    v2 = eigenvectors[:, 1]
    fig.add_trace(go.Scatter(
        x=[0, v2[0]], y=[0, v2[1]],
        mode='lines+markers',
        name=f'v₂ (λ={eigenvalues[1]:.2f})',
        line=dict(color='red', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        xaxis=dict(range=[-2, 2], zeroline=True, zerolinewidth=2),
        yaxis=dict(range=[-2, 2], zeroline=True, zerolinewidth=2),
        width=600, height=600,
        showlegend=True
    )
    
    st.plotly_chart(fig)

elif topic == "線形変換":
    st.header("線形変換の可視化")
    
    # 変換行列の選択
    transform_type = st.selectbox(
        "変換の種類を選択",
        ["カスタム", "回転", "拡大縮小", "せん断"]
    )
    
    if transform_type == "カスタム":
        col1, col2 = st.columns(2)
        with col1:
            t11 = st.number_input("T₁₁", value=2.0, key="t11")
            t12 = st.number_input("T₁₂", value=0.0, key="t12")
        with col2:
            t21 = st.number_input("T₂₁", value=0.0, key="t21")
            t22 = st.number_input("T₂₂", value=1.0, key="t22")
        T = np.array([[t11, t12], [t21, t22]])
    
    elif transform_type == "回転":
        angle = st.slider("回転角度 (度)", -180, 180, 45)
        rad = np.radians(angle)
        T = np.array([[np.cos(rad), -np.sin(rad)], 
                      [np.sin(rad), np.cos(rad)]])
    
    elif transform_type == "拡大縮小":
        sx = st.slider("X方向の倍率", 0.1, 3.0, 1.5)
        sy = st.slider("Y方向の倍率", 0.1, 3.0, 1.0)
        T = np.array([[sx, 0], [0, sy]])
    
    else:  # せん断
        shx = st.slider("X方向のせん断", -2.0, 2.0, 0.5)
        shy = st.slider("Y方向のせん断", -2.0, 2.0, 0.0)
        T = np.array([[1, shx], [shy, 1]])
    
    st.write("変換行列T:")
    st.write(pd.DataFrame(T, columns=['列1', '列2'], index=['行1', '行2']))
    
    # グリッドの生成と変換
    x = np.linspace(-3, 3, 7)
    y = np.linspace(-3, 3, 7)
    
    fig = go.Figure()
    
    # 元のグリッド（薄い色）
    for xi in x:
        fig.add_trace(go.Scatter(
            x=[xi, xi], y=[-3, 3],
            mode='lines',
            line=dict(color='lightgray', width=1),
            showlegend=False
        ))
    for yi in y:
        fig.add_trace(go.Scatter(
            x=[-3, 3], y=[yi, yi],
            mode='lines',
            line=dict(color='lightgray', width=1),
            showlegend=False
        ))
    
    # 変換後のグリッド
    for xi in x:
        points = np.array([[xi, yi] for yi in y])
        transformed = points @ T.T
        fig.add_trace(go.Scatter(
            x=transformed[:, 0], y=transformed[:, 1],
            mode='lines',
            line=dict(color='blue', width=2),
            showlegend=False
        ))
    
    for yi in y:
        points = np.array([[xi, yi] for xi in x])
        transformed = points @ T.T
        fig.add_trace(go.Scatter(
            x=transformed[:, 0], y=transformed[:, 1],
            mode='lines',
            line=dict(color='red', width=2),
            showlegend=False
        ))
    
    # 単位ベクトルの表示
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    e1_t = T @ e1
    e2_t = T @ e2
    
    fig.add_trace(go.Scatter(
        x=[0, e1_t[0]], y=[0, e1_t[1]],
        mode='lines+markers',
        line=dict(color='darkblue', width=4),
        name='e₁変換後',
        marker=dict(size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, e2_t[0]], y=[0, e2_t[1]],
        mode='lines+markers',
        line=dict(color='darkred', width=4),
        name='e₂変換後',
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        xaxis=dict(range=[-5, 5], zeroline=True, zerolinewidth=2),
        yaxis=dict(range=[-5, 5], zeroline=True, zerolinewidth=2),
        width=700, height=700,
        showlegend=True
    )
    
    st.plotly_chart(fig)

# フッター
st.markdown("---")
st.markdown("📚 このアプリは線形代数の概念を視覚的に理解するためのデモアプリです。")
st.markdown("💡 各トピックで値を変更して、結果がどのように変わるか観察してみてください。")