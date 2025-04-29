import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal
import math

class OpticalPhaseQuantumSimulator:
    """白色LEDベースの光学位相方式量子演算エミュレーター"""
    
    def __init__(self, num_qubits=2, coherence_factor=0.8):
        """
        Parameters:
        num_qubits (int): シミュレートする量子ビット数
        coherence_factor (float): 光源のコヒーレンス係数（1に近いほど理想的な単色光）
        """
        self.num_qubits = num_qubits
        self.coherence_factor = coherence_factor
        
        # 2^n 次元の状態ベクトル（各量子ビット状態の振幅）
        self.state_vector = np.zeros(2**num_qubits, dtype=complex)
        self.state_vector[0] = 1.0  # |0...0>の初期状態
        
        # 白色LEDの波長分布をモデル化（可視光範囲）
        self.wavelengths = np.linspace(380, 780, 100)  # 380-780nm
        self.intensity = self._generate_white_led_spectrum()
        
        # 光学ノイズモデル
        self.phase_noise = self._calculate_phase_noise()
    
    def _generate_white_led_spectrum(self):
        """白色LEDのスペクトル生成（青色LEDと黄色蛍光体の組み合わせ）"""
        blue_peak = 450  # 青色ピーク (nm)
        yellow_peak = 580  # 黄色ピーク (nm)
        
        blue_intensity = np.exp(-0.5 * ((self.wavelengths - blue_peak) / 30)**2)
        yellow_intensity = np.exp(-0.5 * ((self.wavelengths - yellow_peak) / 60)**2)
        
        # 青色と黄色の組み合わせ
        return 0.4 * blue_intensity + 0.6 * yellow_intensity
    
    def _calculate_phase_noise(self):
        """コヒーレンス因子に基づく位相ノイズの計算"""
        return (1 - self.coherence_factor) * np.pi
    
    def apply_noise(self, operation_matrix):
        """光学ノイズを操作に適用"""
        noise_level = self.phase_noise
        noise_matrix = np.eye(len(operation_matrix), dtype=complex)
        
        for i in range(len(noise_matrix)):
            phase_error = normal(0, noise_level)
            noise_matrix[i, i] = np.exp(1j * phase_error)
            
        return np.dot(noise_matrix, operation_matrix)
    
    def x_gate(self, qubit_idx):
        """X（NOT）ゲート - 位相空間での180度回転に相当"""
        self._apply_single_qubit_gate(qubit_idx, np.array([[0, 1], [1, 0]]))
    
    def h_gate(self, qubit_idx):
        """Hadamardゲート - 位相空間での45度回転と反射の組み合わせ"""
        h_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        self._apply_single_qubit_gate(qubit_idx, h_matrix)
    
    def z_gate(self, qubit_idx):
        """Zゲート - 位相シフト操作に相当"""
        self._apply_single_qubit_gate(qubit_idx, np.array([[1, 0], [0, -1]]))
    
    def y_gate(self, qubit_idx):
        """Yゲート - 位相シフトと回転の組み合わせ"""
        self._apply_single_qubit_gate(qubit_idx, np.array([[0, -1j], [1j, 0]]))
    
    def phase_shift(self, qubit_idx, phi):
        """任意の位相シフトゲート"""
        phase_matrix = np.array([[1, 0], [0, np.exp(1j * phi)]])
        self._apply_single_qubit_gate(qubit_idx, phase_matrix)
    
    def cnot_gate(self, control_idx, target_idx):
        """CNOTゲート - control量子ビットが|1>の場合にtarget量子ビットを反転"""
        n = self.num_qubits
        I = np.eye(2)
        X = np.array([[0, 1], [1, 0]])
        
        # プロジェクション演算子
        P0 = np.array([[1, 0], [0, 0]])
        P1 = np.array([[0, 0], [0, 1]])
        
        # 制御0のプロジェクション
        ops0 = [I] * n
        ops0[control_idx] = P0
        M0 = ops0[0]
        for op in ops0[1:]:
            M0 = np.kron(M0, op)
        
        # 制御1のプロジェクション + ターゲットにXゲート
        ops1 = [I] * n
        ops1[control_idx] = P1
        ops1[target_idx] = X
        M1 = ops1[0]
        for op in ops1[1:]:
            M1 = np.kron(M1, op)
        
        # CNOT演算子
        cnot_matrix = M0 + M1
        
        # ノイズの適用
        noisy_cnot = self.apply_noise(cnot_matrix)
        
        # 状態ベクトルに適用
        self.state_vector = np.dot(noisy_cnot, self.state_vector)
    
    def _apply_single_qubit_gate(self, qubit_idx, gate_matrix):
        """単一量子ビットゲートを適用する内部メソッド"""
        n = self.num_qubits
        ops = [np.eye(2)] * n
        ops[qubit_idx] = gate_matrix
        
        # テンソル積でn量子ビット操作を構築
        operation = ops[0]
        for op in ops[1:]:
            operation = np.kron(operation, op)
        
        # ノイズの適用
        noisy_operation = self.apply_noise(operation)
        
        # 状態ベクトルに適用
        self.state_vector = np.dot(noisy_operation, self.state_vector)
    
    def measure(self, shots=1024):
        """
        量子状態を測定し、結果の分布を返す
        
        Parameters:
        shots (int): 測定回数
        
        Returns:
        dict: 測定結果と出現回数の辞書
        """
        # 確率分布の計算
        probabilities = np.abs(self.state_vector) ** 2
        
        # サンプリング
        outcomes = np.random.choice(2**self.num_qubits, size=shots, p=probabilities)
        
        # 結果をビット文字列に変換して集計
        result_counts = {}
        for outcome in outcomes:
            # ビット文字列に変換（例: 3 → '11', 2 → '10'）
            bit_string = format(outcome, f'0{self.num_qubits}b')
            result_counts[bit_string] = result_counts.get(bit_string, 0) + 1
        
        return result_counts
    
    def visualize_state(self):
        """量子状態の可視化"""
        probabilities = np.abs(self.state_vector) ** 2
        states = [format(i, f'0{self.num_qubits}b') for i in range(2**self.num_qubits)]
        
        plt.figure(figsize=(10, 6))
        plt.bar(states, probabilities)
        plt.xlabel('量子状態')
        plt.ylabel('確率')
        plt.title('量子状態の確率分布')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt
    
    def visualize_optical_spectrum(self):
        """光源スペクトルの可視化"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.wavelengths, self.intensity)
        plt.xlabel('波長 (nm)')
        plt.ylabel('相対強度')
        plt.title('白色LED光源スペクトル')
        plt.grid(True)
        plt.tight_layout()
        return plt
    
    def simulate_interference(self, phase_difference):
        """
        光学干渉パターンのシミュレーション
        
        Parameters:
        phase_difference (float): 光路差による位相差（ラジアン）
        """
        positions = np.linspace(-5, 5, 200)
        patterns = []
        
        for wavelength, intensity in zip(self.wavelengths, self.intensity):
            # 波数の計算
            k = 2 * np.pi / (wavelength * 1e-9)
            
            # 各位置での干渉パターン
            pattern = intensity * (1 + np.cos(k * positions + phase_difference)) / 2
            patterns.append(pattern)
        
        # 全波長からの寄与を合計
        total_pattern = np.sum(patterns, axis=0)
        total_pattern /= np.max(total_pattern)  # 正規化
        
        plt.figure(figsize=(10, 6))
        plt.plot(positions, total_pattern)
        plt.xlabel('位置 (µm)')
        plt.ylabel('相対強度')
        plt.title(f'位相差 {phase_difference:.2f} radの干渉パターン')
        plt.grid(True)
        plt.tight_layout()
        return plt

# 使用例
def quantum_teleportation_example():
    """量子テレポーテーションの実装例"""
    sim = OpticalPhaseQuantumSimulator(num_qubits=3, coherence_factor=0.9)
    
    # |0>状態を準備
    print("初期状態: |000>")
    
    # 1つ目の量子ビットに任意の状態を準備（テレポートしたい状態）
    sim.h_gate(0)
    sim.phase_shift(0, np.pi/4)
    print("送信する状態を準備 (H・Phase(π/4) を量子ビット0に適用)")
    
    # もつれ合いを作成（量子ビット1と2）
    sim.h_gate(1)
    sim.cnot_gate(1, 2)
    print("量子ビット1と2でBellペアを作成")
    
    # テレポーテーションプロトコル
    sim.cnot_gate(0, 1)
    sim.h_gate(0)
    
    # 測定（古典的情報伝達の代わり）
    # 実際のテレポーテーションでは測定結果に基づいて操作を行いますが、
    # ここではシミュレーション用に直接ゲートを適用
    sim.cnot_gate(1, 2)
    sim.z_gate(2)
    
    print("テレポーテーション完了")
    
    # 結果を測定
    result = sim.measure(shots=1024)
    print("測定結果:", result)
    
    # 状態の可視化
    return sim.visualize_state()

# 量子フーリエ変換の例
def quantum_fourier_transform_example(n_qubits=3):
    """量子フーリエ変換の実装例"""
    sim = OpticalPhaseQuantumSimulator(num_qubits=n_qubits, coherence_factor=0.85)
    
    # 初期状態を準備（すべての基底状態の重ね合わせ）
    for i in range(n_qubits):
        sim.h_gate(i)
    
    # QFTの実装
    for i in range(n_qubits):
        for j in range(i):
            # 制御位相回転
            theta = 2 * np.pi / (2**(i-j+1))
            sim.cnot_gate(j, i)
            sim.phase_shift(i, theta)
            sim.cnot_gate(j, i)
        
        # Hadamardゲート
        sim.h_gate(i)
    
    # 量子ビットの順序を逆転（標準的なQFT定義との整合性のため）
    for i in range(n_qubits // 2):
        for j in range(3):  # SWAPゲートをCNOTで構成
            sim.cnot_gate(i, n_qubits-i-1)
            sim.cnot_gate(n_qubits-i-1, i)
            sim.cnot_gate(i, n_qubits-i-1)
    
    # 結果を測定
    result = sim.measure(shots=1024)
    print("量子フーリエ変換の測定結果:", result)
    
    return sim.visualize_state()
