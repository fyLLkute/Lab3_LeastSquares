import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 1. Зчитування даних з CSV [cite: 81, 92]
def load_data(filename):
    df = pd.read_csv(filename)
    return df['Month'].values.astype(float), df['Temp'].values.astype(float)


# 2. Формування матриці B та вектора C для МНК [cite: 26, 27, 81, 134-144]
def form_system(x, y, m):
    n_nodes = len(x)
    A = np.zeros((m + 1, m + 1))
    B_vec = np.zeros(m + 1)

    for k in range(m + 1):
        for l in range(m + 1):
            A[k, l] = np.sum(x ** (k + l))  # b_kl [cite: 26]
        B_vec[k] = np.sum(y * (x ** k))  # c_k [cite: 27]
    return A, B_vec


# 3. Метод Гаусса з вибором головного елемента [cite: 35, 44, 81, 145-154]
def gauss_solve(A, b):
    n = len(b)
    # Прямий хід [cite: 69, 146]
    for k in range(n):
        # Вибір головного елемента по стовпцю [cite: 44, 147]
        max_row = np.argmax(np.abs(A[k:, k])) + k
        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Зворотній хід [cite: 69, 70, 150-154]
    x_sol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x_sol[i] = (b[i] - np.dot(A[i, i + 1:], x_sol[i + 1:])) / A[i, i]
    return x_sol


# 4. Обчислення значення многочлена [cite: 7, 155-159]
def polynomial(x, coeffs):
    y_poly = np.zeros_like(x, dtype=float)
    for i, a in enumerate(coeffs):
        y_poly += a * (x ** i)
    return y_poly


# 5. Обчислення дисперсії [cite: 30, 95, 160-162]
def calculate_variance(y_true, y_approx):
    n = len(y_true)
    return np.sqrt(np.sum((y_true - y_approx) ** 2) / n)


# Головна частина програми
if __name__ == "__main__":
    # Завантаження даних [cite: 127-130]
    months, temps = load_data('data.csv')

    variances = []
    degrees = list(range(1, 11))  # m = 1...10 [cite: 82, 88]

    # Пошук оптимального степеня [cite: 83, 95, 163-170]
    best_m = 1
    min_var = float('inf')
    best_coeffs = None

    print("Степінь m | Дисперсія")
    print("-" * 25)

    for m in degrees:
        A, B_vec = form_system(months, temps, m)
        coeffs = gauss_solve(A.copy(), B_vec.copy())
        y_approx = polynomial(months, coeffs)
        var = calculate_variance(temps, y_approx)
        variances.append(var)

        print(f"{m:9} | {var:.4f}")

        if var < min_var:
            min_var = var
            best_m = m
            best_coeffs = coeffs

    print("-" * 25)
    print(f"Оптимальний степінь полінома: m = {best_m}")

    # 6. Екстраполяція на 3 місяці [cite: 98, 179-182]
    future_months = np.array([25, 26, 27])
    forecast = polynomial(future_months, best_coeffs)
    print(f"Прогноз температури на місяці 25, 26, 27: {np.round(forecast, 2)}")

    # 7. Табуляція похибки з дрібнішим кроком [cite: 81, 86, 87, 97]
    x_fine = np.linspace(months[0], months[-1], 20 * len(months))
    y_fine_approx = polynomial(x_fine, best_coeffs)
    # Для графіку похибки у вузлах [cite: 184-187]
    error = np.abs(temps - polynomial(months, best_coeffs))

    # Побудова графіків [cite: 82, 93, 96, 193]
    plt.figure(figsize=(12, 8))

    # Графік 1: Дані та Апроксимація
    plt.subplot(2, 2, 1)
    plt.scatter(months, temps, color='red', label='Фактичні дані')
    plt.plot(x_fine, y_fine_approx, label=f'МНК Поліном (m={best_m})')
    plt.title('Апроксимація температури')
    plt.legend()

    # Графік 2: Залежність дисперсії від степеня m [cite: 82]
    plt.subplot(2, 2, 2)
    plt.plot(degrees, variances, marker='o', color='green')
    plt.title('Залежність дисперсії від степеня m')
    plt.xlabel('Степінь m')
    plt.ylabel('Дисперсія')

    # Графік 3: Похибка апроксимації [cite: 97]
    plt.subplot(2, 2, 3)
    plt.bar(months, error, color='purple')
    plt.title('Похибка у вузлах |f(x) - phi(x)|')
    plt.xlabel('Місяць')

    # Графік 4: Прогноз
    plt.subplot(2, 2, 4)
    full_months = np.concatenate([months, future_months])
    full_preds = polynomial(full_months, best_coeffs)
    plt.plot(months, temps, 'ro', label='Минуле')
    plt.plot(future_months, forecast, 'bo', label='Прогноз')
    plt.plot(full_months, full_preds, '--k', alpha=0.5)
    plt.title('Прогноз на 3 місяці')
    plt.legend()

    plt.tight_layout()
    plt.show()