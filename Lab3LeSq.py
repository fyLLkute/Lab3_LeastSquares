import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 1. Зчитування даних з CSV
def load_data(filename):
    df = pd.read_csv(filename) # Зчитування з CSV файлу
    temps = df['Temp'].values.astype(float) # Масив f_i

    n = len(temps)  # n = 24 вузли
    x0 = 1  # Перший місяць
    xn = 24  # Останній місяць

    h = (xn - x0) / n

    # Табуляція:  масив вузлів xi
    months = np.array([x0 + i * h for i in range(n)])

    return months, temps

# 2. Формування матриці B та вектора C для МНК
def form_system(x, y, m):
    n_nodes = len(x)
    A = np.zeros((m + 1, m + 1))
    B_vec = np.zeros(m + 1)

    for k in range(m + 1):
        for l in range(m + 1):
            A[k, l] = np.sum(x ** (k + l))  # b_kl
        B_vec[k] = np.sum(y * (x ** k))  # c_k
    return A, B_vec


# 3. Метод Гаусса з вибором головного елемента
def gauss_solve(A, b):
    n = len(b)
    # Прямий хід
    for k in range(n):
        # Вибір головного елемента по стовпцю
        max_row = np.argmax(np.abs(A[k:, k])) + k
        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Зворотній хід
    x_sol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x_sol[i] = (b[i] - np.dot(A[i, i + 1:], x_sol[i + 1:])) / A[i, i]
    return x_sol


# 4. Обчислення значення многочлена
def polynomial(x, coeffs):
    y_poly = np.zeros_like(x, dtype=float)
    for i, a in enumerate(coeffs):
        y_poly += a * (x ** i)
    return y_poly


# 5. Обчислення дисперсії
def calculate_variance(y_true, y_approx):
    n = len(y_true)
    return np.sqrt(np.sum((y_true - y_approx) ** 2) / (n + 1) )


# Головна частина програми
if __name__ == "__main__":
    # Завантаження даних
    months, temps = load_data('data.csv')

    variances = []
    # Високі степені (5-10) спричиняють різкі стрибки при екстраполяції
    degrees = list(range(1, 4))

    best_m = 1
    min_var = float('inf')
    best_coeffs = None

    print(f"Аналіз даних за {len(months)} періодів")
    print("Степінь m | Дисперсія")
    print("-" * 25)

    for m in degrees:
        A, B_vec = form_system(months, temps, m) #  Формування системи рівнянь для поточного степеня m
        coeffs = gauss_solve(A.copy(), B_vec.copy()) # Розв'язання системи (знаходження коефіцієнтів a_i)
        y_approx = polynomial(months, coeffs) # Побудова значень многочлена для подальшого аналізу
   # Обчислення дисперсії для кожного випадку
        var = calculate_variance(temps, y_approx)
        variances.append(var)

        print(f"{m:9} | {var:.4f}")

        # Вибір оптимального m за мінімумом дисперсії
        if var < min_var:
            min_var = var
            best_m = m
            best_coeffs = coeffs

    print("-" * 25)
    print(f"Оптимальний стабільний степінь: m = {best_m}")

    # 6. Екстраполяція на 3 наступні періоди
    # Робимо прогноз автоматично від останнього дня у файлі
    last_val = months[-1]
    future_months = np.array([last_val + 1, last_val + 2, last_val + 3])

    forecast = polynomial(future_months, best_coeffs)
    print(f"Прогноз температури на наступні кроки {future_months}: {np.round(forecast, 2)}")

    x0 = months[0]
    xn = months[-1]
    n_count = len(months)

    # Розрахунок кроку h1 точно за формулою  h1 = (xn - x0) / (20 * n)
    h1 = (xn - x0) / (20 * n_count)

    # Створення дрібної сітки вузлів
    x_fine_list = []
    for i in range(20 * n_count + 1):
        xi = x0 + i * h1
        x_fine_list.append(xi)

    x_fine = np.array(x_fine_list)

    # Обчислення значень многочлена для цих нових точок
    y_fine_approx = polynomial(x_fine, best_coeffs)
    # Для графіку похибки у вузлах
    error = np.abs(temps - polynomial(months, best_coeffs))

    # Побудова графіків
    plt.figure(figsize=(12, 8))

    # Графік 1: Дані та Апроксимація
    plt.subplot(2, 2, 1)
    plt.scatter(months, temps, color='red', label='Фактичні дані')
    plt.plot(x_fine, y_fine_approx, label=f'МНК Поліном (m={best_m})')
    plt.title('Апроксимація температури')
    plt.legend()

    # Графік 2: Залежність дисперсії від степеня m
    plt.subplot(2, 2, 2)
    plt.plot(degrees, variances, marker='o', color='green')
    plt.title('Залежність дисперсії від степеня m')
    plt.xlabel('Степінь m')
    plt.ylabel('Дисперсія')

    # Графік 3: Похибка апроксимації
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