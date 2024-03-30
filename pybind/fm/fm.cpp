#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <utility>
#include <queue>
#include <iostream>

namespace py = pybind11;

template<class T>
T Sign(const T phi)
{
    return phi <= T(0) ? T(-1) : T(1);
}

template<class T>
bool Interface(const T phi1, const T phi2)
{
    return Sign(phi1) != Sign(phi2);
}

template<class T>
T Theta(const T phi1, const T phi2)
{
    return phi1 / (phi1 - phi2);
}

template<class T>
bool solve_quadratic(const T p1, const T p2, const T dx, T& rst)
{
    if (abs(p1) >= abs(p2) + dx)
    {
        rst = p2 + dx;
        return true;
    }
    else if (abs(p2) >= abs(p1) + dx)
    {
        rst = p1 + dx;
        return true;
    }
    else
    {
        T delta = T(2) * dx * dx - pow(p1 - p2, 2);
        if (delta < T(0))
        {
            std::cerr << "Error: [Levelset] negative delta in Solve_Quadratic_2" << std::endl; 
            return false;
        }
        rst = T(.5) * (p1 + p2 + sqrt(delta));
        return true;
    }
}

template<class T>
bool solve_quadratic(const T p1, const T p2, const T p3, const T dx, T& rst)
{
    T delta = pow(p1 + p2 + p3, 2) - (T)3 * (p1 * p1 + p2 * p2 + p3 * p3 - dx * dx);
    if (delta < (T)0) {
        int i = 0;
        T p_max = abs(p1);
        if (abs(p2) > p_max)
        {
            i = 1;
            p_max = abs(p2);
        }
        if (abs(p3) > p_max)
        {
            i = 2;
            p_max = abs(p3);
        }
        T q1, q2; 
        if (i == 0) 
        {
            q1 = p2;
            q2 = p3;
        }
        else if (i == 1)
        {
            q1 = p1;
            q2 = p3;
        }
        else
        {
            q1 = p1;
            q2 = p2;
        }
        return solve_quadratic(q1, q2, dx, rst);
    }
    rst = (T)1 / (T)3 * (p1 + p2 + p3 + sqrt(delta)); 
    return true;
}
template<class T>
T solve_eikonal_2d(const int i, const int j, const int res_x, const int res_y, const T dx, const std::vector<T>& tent, const std::vector<unsigned short>& done)
{
    T correct_phi_x = std::numeric_limits<T>::max();
    T correct_phi_y = std::numeric_limits<T>::max();
    int correct_axis_x = 0;
    int correct_axis_y = 0;
    if (i > 0)
    {
        int nb_idx = (i - 1) * res_y + j;
        if (done[nb_idx])
        {
            correct_phi_x = std::min(tent[nb_idx], correct_phi_x);
            correct_axis_x = 1;
        }
    }
    if (i < res_x - 1)
    {
        int nb_idx = (i + 1) * res_y + j;
        if (done[nb_idx])
        {
            correct_phi_x = std::min(tent[nb_idx], correct_phi_x);
            correct_axis_x = 1;
        }
    }
    if (j > 0)
    {
        int nb_idx = i * res_y + j - 1;
        if (done[nb_idx])
        {
            correct_phi_y = std::min(tent[nb_idx], correct_phi_y);
            correct_axis_y = 1;
        }
    }
    if (j < res_y - 1)
    {
        int nb_idx = i * res_y + j + 1;
        if (done[nb_idx])
        {
            correct_phi_y = std::min(tent[nb_idx], correct_phi_y);
            correct_axis_y = 1;
        }
    }
    T new_phi;
    int n = correct_axis_x + correct_axis_y;
    switch (n)
    {
    case 1: {
        T c_phi;
        if (correct_axis_x)
            c_phi = correct_phi_x;
        else
            c_phi = correct_phi_y;
        new_phi = c_phi + dx;
    }break;
    case 2:
    {
        solve_quadratic(correct_phi_x, correct_phi_y, dx, new_phi);
    }break;
    default: {
        std::cerr << "Error: [Levelset] bad solving Eikonal" << std::endl;
        std::exit(1);
    }break;
    }
    return new_phi;
}

template<class T>
T solve_eikonal_3d(const int i, const int j, const int k, const int res_x, const int res_y, const int res_z, const T dx, const std::vector<T>& tent, const std::vector<unsigned short>& done)
{
    T correct_phi_x = std::numeric_limits<T>::max();
    T correct_phi_y = std::numeric_limits<T>::max();
    T correct_phi_z = std::numeric_limits<T>::max();
    int correct_axis_x = 0;
    int correct_axis_y = 0;
    int correct_axis_z = 0;

    if (i > 0)
    {
        int nb_idx = (i - 1) * res_y * res_z + j * res_z + k;
        if (done[nb_idx])
        {
            correct_phi_x = std::min(tent[nb_idx], correct_phi_x);
            correct_axis_x = 1;
        }
    }
    if (i < res_x - 1)
    {
        int nb_idx = (i + 1) * res_y * res_z + j * res_z + k;
        if (done[nb_idx])
        {
            correct_phi_x = std::min(tent[nb_idx], correct_phi_x);
            correct_axis_x = 1;
        }
    }
    if (j > 0)
    {
        int nb_idx = i * res_y * res_z + (j - 1) * res_z + k;
        if (done[nb_idx])
        {
            correct_phi_y = std::min(tent[nb_idx], correct_phi_y);
            correct_axis_y = 1;
        }
    }
    if (j < res_y - 1)
    {
        int nb_idx = i * res_y * res_z + (j + 1) * res_z + k;
        if (done[nb_idx])
        {
            correct_phi_y = std::min(tent[nb_idx], correct_phi_y);
            correct_axis_y = 1;
        }
    }
    if (k > 0)
    {
        int nb_idx = i * res_y * res_z + j * res_z + k - 1;
        if (done[nb_idx])
        {
            correct_phi_z = std::min(tent[nb_idx], correct_phi_z);
            correct_axis_z = 1;
        }
    }
    if (k < res_z - 1)
    {
        int nb_idx = i * res_y * res_z + j * res_z + k + 1;
        if (done[nb_idx])
        {
            correct_phi_z = std::min(tent[nb_idx], correct_phi_z);
            correct_axis_z = 1;
        }
    }
    T new_phi;
    int n = correct_axis_x + correct_axis_y + correct_axis_z;
    switch (n)
    {
    case 1: {
        T c_phi;
        if (correct_axis_x)
            c_phi = correct_phi_x;
        else if (correct_axis_y)
            c_phi = correct_phi_y;
        else
            c_phi = correct_phi_z;
        new_phi = c_phi + dx;
    }break;
    case 2:
    {
        if (!correct_axis_z)
            solve_quadratic(correct_phi_x, correct_phi_y, dx, new_phi);
        else if(!correct_axis_y)
            solve_quadratic(correct_phi_x, correct_phi_z, dx, new_phi);
        else
            solve_quadratic(correct_phi_y, correct_phi_z, dx, new_phi);
    }break;
    case 3:
    {
        solve_quadratic(correct_phi_x, correct_phi_y, correct_phi_z, dx, new_phi);
    }break;
    default: {
        std::cerr << "Error: [Levelset] bad solving Eikonal" << std::endl;
        std::exit(1);
    }break;
    }
    return new_phi;
}

template<class T>
void fm_2d(py::array_t<T>& phi, const T band_width, const T dx)
{
    auto phi_acc = phi.mutable_unchecked<2>();
    int res_x = phi.shape(0);
    int res_y = phi.shape(1);

    const int cell_num = res_x * res_y;
    std::vector<T> tent(cell_num, band_width < 0?std::numeric_limits<T>::max():band_width);  
    std::vector<unsigned short> done(cell_num, 0);
    using PRI = std::pair<T, int>;
    std::priority_queue<PRI, std::vector<PRI>, std::greater<PRI> > heaps[2];

    // precondition
    // find interface cells
    for (int i = 0; i < res_x; i++)
        for (int j = 0; j < res_y; j++)
        {
            int idx = i * res_y + j;
            if (i > 0)
                if (Interface(phi_acc(i, j), phi_acc(i - 1, j)))
                {

                    done[idx] = 1;
                    continue;
                }
            if(i < res_x - 1)
                if (Interface(phi_acc(i, j), phi_acc(i + 1, j)))
                {
                    done[idx] = 1;
                    continue;
                }
            if (j > 0)
                if (Interface(phi_acc(i, j), phi_acc(i, j - 1)))
                {
                    done[idx] = 1;
                    continue;
                }
            if(j < res_y - 1)
                if (Interface(phi_acc(i, j), phi_acc(i, j + 1)))
                {
                    done[idx] = 1;
                    continue;
                }
        }

    // calculate interface phi values
    for (int i = 0; i < res_x; i++)
        for (int j = 0; j < res_y; j++)
        {
            int idx = i * res_y + j;
            if (!done[idx])
                continue;
            T correct_phi_x = std::numeric_limits<T>::max();
            T correct_phi_y = std::numeric_limits<T>::max();
            int correct_axis_x = 0;
            int correct_axis_y = 0;
            if (i > 0)
            {
                int nb_idx = (i - 1) * res_y + j;
                if (done[nb_idx] && Interface(phi_acc(i, j), phi_acc(i - 1, j)))
                {
                    T c_phi = Theta(phi_acc(i, j), phi_acc(i - 1, j)) * dx;
                    correct_phi_x = std::min(c_phi, correct_phi_x);
                    correct_axis_x = 1;
                }
            }
            if (i < res_x - 1)
            {
                int nb_idx = (i + 1) * res_y + j;
                if (done[nb_idx] && Interface(phi_acc(i, j), phi_acc(i + 1, j)))
                {
                    T c_phi = Theta(phi_acc(i, j), phi_acc(i + 1, j)) * dx;
                    correct_phi_x = std::min(c_phi, correct_phi_x);
                    correct_axis_x = 1;
                }
            }
            if (j > 0)
            {
                int nb_idx = i * res_y + j - 1;
                if (done[nb_idx] && Interface(phi_acc(i, j), phi_acc(i, j - 1)))
                {
                    T c_phi = Theta(phi_acc(i, j), phi_acc(i, j - 1)) * dx;
                    correct_phi_y = std::min(c_phi, correct_phi_y);
                    correct_axis_y = 1;
                }
            }
            if (j < res_y - 1)
            {
                int nb_idx = i * res_y + j + 1;
                if (done[nb_idx] && Interface(phi_acc(i, j), phi_acc(i, j + 1)))
                {
                    T c_phi = Theta(phi_acc(i, j), phi_acc(i, j + 1)) * dx;
                    correct_phi_y = std::min(c_phi, correct_phi_y);
                    correct_axis_y = 1;
                }
            }

            if (correct_axis_x || correct_axis_y)
            {
                T hmnc_mean = T(0);
                if (correct_axis_x)
                    hmnc_mean += T(1) / (correct_phi_x * correct_phi_x);
                if (correct_axis_y)
                    hmnc_mean += T(1) / (correct_phi_y * correct_phi_y);
                hmnc_mean = sqrt(T(1) / hmnc_mean);
                tent[idx] = hmnc_mean;
            }
            else
            {
                std::cerr << "Error: [Levelset] bad preconditioning" << std::endl;
                std::exit(1);
            }
        }

    for (int i = 0; i < res_x; i++)
        for (int j = 0; j < res_y; j++)
        {
            int idx = i * res_y + j;
            bool is_front = false;
            if (i > 0)
            {
                int nb_idx = (i - 1) * res_y + j;
                if (done[nb_idx])
                    is_front = true;
            }
            if (i <res_x - 1)
            {
                int nb_idx = (i + 1) * res_y + j;
                if (done[nb_idx])
                    is_front = true;
            }
            if (j > 0)
            {
                int nb_idx = i * res_y + j - 1;
                if (done[nb_idx])
                    is_front = true;
            }
            if (j < res_y - 1)
            {
                int nb_idx = i * res_y + j + 1;
                if (done[nb_idx])
                    is_front = true;
            }

            if (is_front)
            {
                T temp = solve_eikonal_2d(i, j, res_x, res_y, dx, tent, done);
                if (temp < tent[idx])
                {
                    tent[idx] = temp;
                    heaps[Sign(phi_acc(i, j)) > 0 ? 0 : 1].push(PRI(temp, idx));
                }
            }
        }
    // heap traversing
    for (int h = 0; h < 2; h++)
    {
        auto& heap = heaps[h];
        while (!heap.empty())
        {
            const T top_val = heap.top().first;
            const int idx = heap.top().second;
            heap.pop();
            int i = idx / res_y;
            int j = idx % res_y;
            if (tent[idx] != top_val)
                continue;
            done[idx] = 1;
            if (i > 0)
            {
                int nb_idx = (i - 1) * res_y + j;
                if (!done[nb_idx])
                {
                    T temp = solve_eikonal_2d(i - 1, j, res_x, res_y, dx, tent, done);
                    if (temp < tent[nb_idx])
                    {
                        tent[nb_idx] = temp;
                        heap.push(PRI(temp, nb_idx));
                    }
                }
            }
            if (i < res_x - 1)
            {
                int nb_idx = (i + 1) * res_y + j;
                if (!done[nb_idx])
                {
                    T temp = solve_eikonal_2d(i + 1, j, res_x, res_y, dx, tent, done);
                    if (temp < tent[nb_idx])
                    {
                        tent[nb_idx] = temp;
                        heap.push(PRI(temp, nb_idx));
                    }
                }
            }
            if (j > 0)
            {
                int nb_idx = i * res_y + j - 1;
                if (!done[nb_idx])
                {
                    T temp = solve_eikonal_2d(i, j - 1, res_x, res_y, dx, tent, done);
                    if (temp < tent[nb_idx])
                    {
                        tent[nb_idx] = temp;
                        heap.push(PRI(temp, nb_idx));
                    }
                }
            }
            if (j < res_y - 1)
            {
                int nb_idx = i * res_y + j + 1;
                if (!done[nb_idx])
                {
                    T temp = solve_eikonal_2d(i, j + 1, res_x, res_y, dx, tent, done);
                    if (temp < tent[nb_idx])
                    {
                        tent[nb_idx] = temp;
                        heap.push(PRI(temp, nb_idx));
                    }
                }
            }
        }
    }
    for (int i = 0; i < res_x; i++)
        for (int j = 0; j < res_y; j++)
        {
            int idx = i * res_y + j;
            phi_acc(i, j) = Sign(phi_acc(i, j)) * tent[idx];
        }
}

template<class T>
void fm_3d(py::array_t<T>& phi, const T band_width, const T dx)
{
    auto phi_acc = phi.mutable_unchecked<3>();
    int res_x = phi.shape(0);
    int res_y = phi.shape(1);
    int res_z = phi.shape(2);

    const int cell_num = res_x * res_y * res_z;
    std::vector<T> tent(cell_num, band_width < 0 ? std::numeric_limits<T>::max() : band_width);
    std::vector<unsigned short> done(cell_num, 0);
    using PRI = std::pair<T, int>;
    std::priority_queue<PRI, std::vector<PRI>, std::greater<PRI> > heaps[2];

    // precondition
    // find interface cells
    for (int i = 0; i < res_x; i++)
        for (int j = 0; j < res_y; j++)
            for (int k = 0; k < res_z; k++)
            {
                int idx = i * res_y * res_z + j * res_z + k;
                if (i > 0)
                    if (Interface(phi_acc(i, j, k), phi_acc(i - 1, j, k)))
                    {
                        done[idx] = 1;
                        continue;
                    }
                if (i < res_x - 1)
                    if (Interface(phi_acc(i, j, k), phi_acc(i + 1, j, k)))
                    {
                        done[idx] = 1;
                        continue;
                    }
                if (j > 0)
                    if (Interface(phi_acc(i, j, k), phi_acc(i, j - 1, k)))
                    {
                        done[idx] = 1;
                        continue;
                    }
                if (j < res_y - 1)
                    if (Interface(phi_acc(i, j, k), phi_acc(i, j + 1, k)))
                    {
                        done[idx] = 1;
                        continue;
                    }
                if (k > 0)
                    if (Interface(phi_acc(i, j, k), phi_acc(i, j, k - 1)))
                    {
                        done[idx] = 1;
                        continue;
                    }
                if (k < res_z - 1)
                    if (Interface(phi_acc(i, j, k), phi_acc(i, j, k + 1)))
                    {
                        done[idx] = 1;
                        continue;
                    }
            }

    // calculate interface phi values
    for (int i = 0; i < res_x; i++)
        for (int j = 0; j < res_y; j++)
            for (int k = 0; k < res_z; k++)
            {
                int idx = i * res_y * res_z + j * res_z + k;
                if (!done[idx])
                    continue;
                T correct_phi_x = std::numeric_limits<T>::max();
                T correct_phi_y = std::numeric_limits<T>::max();
                T correct_phi_z = std::numeric_limits<T>::max();
                int correct_axis_x = 0;
                int correct_axis_y = 0;
                int correct_axis_z = 0;
                if (i > 0)
                {
                    int nb_idx = (i - 1) * res_y * res_z + j * res_z + k;
                    if (done[nb_idx] && Interface(phi_acc(i, j, k), phi_acc(i - 1, j, k)))
                    {
                        T c_phi = Theta(phi_acc(i, j, k), phi_acc(i - 1, j, k)) * dx;
                        correct_phi_x = std::min(c_phi, correct_phi_x);
                        correct_axis_x = 1;
                    }
                }
                if (i < res_x - 1)
                {
                    int nb_idx = (i + 1) * res_y * res_z + j * res_z + k;
                    if (done[nb_idx] && Interface(phi_acc(i, j, k), phi_acc(i + 1, j, k)))
                    {
                        T c_phi = Theta(phi_acc(i, j, k), phi_acc(i + 1, j, k)) * dx;
                        correct_phi_x = std::min(c_phi, correct_phi_x);
                        correct_axis_x = 1;
                    }
                }
                if (j > 0)
                {
                    int nb_idx = i * res_y * res_z + (j - 1) * res_z + k;
                    if (done[nb_idx] && Interface(phi_acc(i, j, k), phi_acc(i, j - 1, k)))
                    {
                        T c_phi = Theta(phi_acc(i, j, k), phi_acc(i, j - 1, k)) * dx;
                        correct_phi_y = std::min(c_phi, correct_phi_y);
                        correct_axis_y = 1;
                    }
                }
                if (j < res_y - 1)
                {
                    int nb_idx = i * res_y * res_z + (j + 1) * res_z + k;
                    if (done[nb_idx] && Interface(phi_acc(i, j, k), phi_acc(i, j + 1, k)))
                    {
                        T c_phi = Theta(phi_acc(i, j, k), phi_acc(i, j + 1, k)) * dx;
                        correct_phi_y = std::min(c_phi, correct_phi_y);
                        correct_axis_y = 1;
                    }
                }
                if (k > 0)
                {
                    int nb_idx = i * res_y * res_z + j * res_z + k - 1;
                    if (done[nb_idx] && Interface(phi_acc(i, j, k), phi_acc(i, j, k - 1)))
                    {
                        T c_phi = Theta(phi_acc(i, j, k), phi_acc(i, j, k - 1)) * dx;
                        correct_phi_z = std::min(c_phi, correct_phi_z);
                        correct_axis_z = 1;
                    }
                }
                if (k < res_z - 1)
                {
                    int nb_idx = i * res_y * res_z + j * res_z + k + 1;
                    if (done[nb_idx] && Interface(phi_acc(i, j, k), phi_acc(i, j, k + 1)))
                    {
                        T c_phi = Theta(phi_acc(i, j, k), phi_acc(i, j, k + 1)) * dx;
                        correct_phi_z = std::min(c_phi, correct_phi_z);
                        correct_axis_z = 1;
                    }
                }
                if (correct_axis_x || correct_axis_y || correct_axis_z)
                {
                    T hmnc_mean = T(0);
                    if (correct_axis_x)
                        hmnc_mean += T(1) / (correct_phi_x * correct_phi_x);
                    if (correct_axis_y)
                        hmnc_mean += T(1) / (correct_phi_y * correct_phi_y);
                    if (correct_axis_z)
                        hmnc_mean += T(1) / (correct_phi_z * correct_phi_z);
                    hmnc_mean = sqrt(T(1) / hmnc_mean);
                    tent[idx] = hmnc_mean;
                }
                else
                {
                    std::cerr << "Error: [Levelset] bad preconditioning" << std::endl;
                    std::exit(1);
                }
            }

    for (int i = 0; i < res_x; i++)
        for (int j = 0; j < res_y; j++)
            for (int k = 0; k < res_z; k++)
            {
                int idx = i * res_y * res_z + j * res_z + k;
                bool is_front = false; 
                if (i > 0)
                {
                    int nb_idx = (i - 1) * res_y * res_z + j * res_z + k;
                    if (done[nb_idx])
                        is_front = true;
                }
                if (i < res_x - 1)
                {
                    int nb_idx = (i + 1) * res_y * res_z + j * res_z + k;
                    if (done[nb_idx])
                        is_front = true;
                }
                if (j > 0)
                {
                    int nb_idx = i * res_y * res_z + (j - 1) * res_z + k;
                    if (done[nb_idx])
                        is_front = true;
                }
                if (j < res_y - 1)
                {
                    int nb_idx = i * res_y * res_z + (j + 1) * res_z + k;
                    if (done[nb_idx])
                        is_front = true;
                }
                if (k > 0)
                {
                    int nb_idx = i * res_y * res_z + j * res_z + k - 1;
                    if (done[nb_idx])
                        is_front = true;
                }
                if (k < res_z - 1)
                {
                    int nb_idx = i * res_y * res_z + j * res_z + k + 1;
                    if (done[nb_idx])
                        is_front = true;
                }
                if (is_front)
                {
                    T temp = solve_eikonal_3d(i, j, k, res_x, res_y, res_z, dx, tent, done);
                    if (temp < tent[idx])
                    {
                        tent[idx] = temp;
                        heaps[Sign(phi_acc(i, j, k)) > 0 ? 0 : 1].push(PRI(temp, idx));
                    }
                }
            }
    // heap traversing
    for (int h = 0; h < 2; h++)
    {
        auto& heap = heaps[h];
        while (!heap.empty())
        {
            const T top_val = heap.top().first;
            const int idx = heap.top().second;
            heap.pop();
            int i = idx / (res_y * res_z);
            int j = (idx / res_z) % res_y;
            int k = idx % res_z;
            if (tent[idx] != top_val)
                continue;
            done[idx] = 1;
            if (i > 0)
            {
                int nb_idx = (i - 1) * res_y * res_z + j * res_z + k;
                if (!done[nb_idx])
                {
                    T temp = solve_eikonal_3d(i - 1, j, k, res_x, res_y, res_z, dx, tent, done);
                    if (temp < tent[nb_idx])
                    {
                        tent[nb_idx] = temp;
                        heap.push(PRI(temp, nb_idx));
                    }
                }
            }
            if (i < res_x - 1)
            {
                int nb_idx = (i + 1) * res_y * res_z + j * res_z + k;
                if (!done[nb_idx])
                {
                    T temp = solve_eikonal_3d(i + 1, j, k, res_x, res_y, res_z, dx, tent, done);
                    if (temp < tent[nb_idx])
                    {
                        tent[nb_idx] = temp;
                        heap.push(PRI(temp, nb_idx));
                    }
                }
            }
            if (j > 0)
            {
                int nb_idx = i * res_y * res_z + (j - 1) * res_z + k;
                if (!done[nb_idx])
                {
                    T temp = solve_eikonal_3d(i, j - 1, k, res_x, res_y, res_z, dx, tent, done);
                    if (temp < tent[nb_idx])
                    {
                        tent[nb_idx] = temp;
                        heap.push(PRI(temp, nb_idx));
                    }
                }
            }
            if (j < res_y - 1)
            {
                int nb_idx = i * res_y * res_z + (j + 1) * res_z + k;
                if (!done[nb_idx])
                {
                    T temp = solve_eikonal_3d(i, j + 1, k, res_x, res_y, res_z, dx, tent, done);
                    if (temp < tent[nb_idx])
                    {
                        tent[nb_idx] = temp;
                        heap.push(PRI(temp, nb_idx));
                    }
                }
            }
            if (k > 0)
            {
                int nb_idx = i * res_y * res_z + j * res_z + k - 1;
                if (!done[nb_idx])
                {
                    T temp = solve_eikonal_3d(i, j, k - 1, res_x, res_y, res_z, dx, tent, done);
                    if (temp < tent[nb_idx])
                    {
                        tent[nb_idx] = temp;
                        heap.push(PRI(temp, nb_idx));
                    }
                }
            }
            if (k < res_z - 1)
            {
                int nb_idx = i * res_y * res_z + j * res_z + k + 1;
                if (!done[nb_idx])
                {
                    T temp = solve_eikonal_3d(i, j, k + 1, res_x, res_y, res_z, dx, tent, done);
                    if (temp < tent[nb_idx])
                    {
                        tent[nb_idx] = temp;
                        heap.push(PRI(temp, nb_idx));
                    }
                }
            }
        }
    }
    for (int i = 0; i < res_x; i++)
        for (int j = 0; j < res_y; j++)
            for (int k = 0; k < res_z; k++)
            {
                int idx = i * res_y * res_z + j * res_z + k;
                phi_acc(i, j, k) = Sign(phi_acc(i, j, k)) * tent[idx];
            }

}

PYBIND11_MODULE(fm, m)
{
    m.doc() = "fast marching"; // optional module docstring
    m.def("fm_2d", py::overload_cast<py::array_t<float> &, const float, const float>(&fm_2d<float>), "fast marching in 2d for float");
    m.def("fm_2d", py::overload_cast<py::array_t<double> &, const double, const double>(&fm_2d<double>), "fast marching in 2d for double");
    m.def("fm_3d", py::overload_cast<py::array_t<float> &, const float, const float>(&fm_3d<float>), "fast marching in 3d for float");
    m.def("fm_3d", py::overload_cast<py::array_t<double> &, const double, const double>(&fm_3d<double>), "fast marching in 3d for double");
}