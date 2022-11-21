#include "stdio.h"
#include "stdlib.h"
#include "stdint.h"
#include "string"
#include "time.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "fstream"

#include <vector>

#define input_layer_neuron 900
#define first_layer_neuron 100
#define second_layer_neuron 50
#define output_layer_neuron 5

int correct_guess = 0;
int num_guess = 0;

int used_epoch = 10000;

std::vector<int> correct_guesses(used_epoch);
std::vector<int> epochs(used_epoch);
std::vector<double> accuracy(used_epoch);

typedef struct
{
    double **data;
    int row;
    int collumn;
} matrix_t;

matrix_t *create_matrix(int row, int collumn);
void print_matrix(matrix_t *mat);
void fill_matrix(matrix_t *mat, double data);

matrix_t *multiply_matrix(matrix_t *mat1, matrix_t *mat2);
matrix_t *add_matrix(matrix_t *mat1, matrix_t *mat2);
matrix_t *subtract_matrix(matrix_t *mat1, matrix_t *mat2);
matrix_t *transpose_matrix(matrix_t *mat);
matrix_t *import_img(char *addr);
matrix_t *import_img_1d(char *addr);

void multiply_matrix(matrix_t *mat1, matrix_t *mat2, matrix_t *ret_mat);
void add_matrix(matrix_t *mat1, matrix_t *mat2, matrix_t *ret_mat);
void subtract_matrix(matrix_t *mat1, matrix_t *mat2, matrix_t *ret_mat);
void transpose_matrix(matrix_t *mat, matrix_t *ret_mat);
void import_img(char *addr, matrix_t *ret_mat);
void import_img_1d(char *addr, matrix_t *ret_mat);
void divide_matrix_scalar(matrix_t *mat, double divider);

void normalize_matrix(matrix_t *mat, double in_min, double in_max, double out_min, double out_max);
void load_param(matrix_t *w1, matrix_t *b1, matrix_t *w2, matrix_t *b2, matrix_t *w3, matrix_t *b3);
void save_param(matrix_t *w1, matrix_t *b1, matrix_t *w2, matrix_t *b2, matrix_t *w3, matrix_t *b3);
void init_rand(matrix_t *w1, matrix_t *b1, matrix_t *w2, matrix_t *b2, matrix_t *w3, matrix_t *b3);

matrix_t *sigmoid_tf(matrix_t *mat);
matrix_t *relu_tf(matrix_t *mat);
matrix_t *sigmoid_dev(matrix_t *mat);
matrix_t *relu_dev(matrix_t *mat);
matrix_t *create_output(int correct_num);
matrix_t *copy_matrix(matrix_t *mat);

void sigmoid_tf(matrix_t *mat, matrix_t *ret_mat);
void relu_tf(matrix_t *mat, matrix_t *ret_mat);
void sigmoid_dev(matrix_t *mat, matrix_t *ret_mat);
void relu_dev(matrix_t *mat, matrix_t *ret_mat);
void create_output(int correct_num, matrix_t *ret_mat);
void copy_matrix(matrix_t *mat, matrix_t *ret_mat);

void print_matrix_size(matrix_t *mat);
matrix_t *multiply_matrix_same_size(matrix_t *mat1, matrix_t *mat2);
void multiply_matrix_same_size(matrix_t *mat1, matrix_t *mat2, matrix_t *ret_mat);
void gradient_descent(matrix_t *mat, matrix_t *mat_correction, double learning_rate);
void gradient_descent(matrix_t *mat, matrix_t *mat_correction, double learning_rate, matrix_t *ret_mat);
matrix_t *gradient_descent_ret(matrix_t *mat, matrix_t *mat_correction, double learning_rate);

double highest_output(matrix_t *mat);
void print_output(matrix_t *mat);

matrix_t *create_matrix(int row, int collumn)
{
    matrix_t *mat = (matrix_t *)malloc(sizeof(matrix_t));
    mat->row = row;
    mat->collumn = collumn;

    // data[row]
    mat->data = (double **)malloc(row * sizeof(double));

    for (int i = 0; i < row; i++)
    {
        mat->data[i] = (double *)malloc(collumn * sizeof(double));
    }

    return mat;
}

void print_matrix(matrix_t *mat)
{
    for (int rows = 0; rows < mat->row; rows++)
    {
        for (int cols = 0; cols < mat->collumn; cols++)
        {
            if (mat->data[rows][cols] >= 0)
                printf("%.2f  ", mat->data[rows][cols]);
            else
                printf("%.1f  ", mat->data[rows][cols]);
        }
        printf("\n");
    }
}
void fill_matrix(matrix_t *mat, double data)
{
    for (int rows = 0; rows < mat->row; rows++)
    {
        for (int cols = 0; cols < mat->collumn; cols++)
        {
            mat->data[rows][cols] = data;
        }
    }
}

matrix_t *multiply_matrix(matrix_t *mat1, matrix_t *mat2)
{
    if (mat1->collumn != mat2->row)
    {
        printf("multiply matrix size error\n");
        fflush(stdout);
    }

    matrix_t *mat = create_matrix(mat1->row, mat2->collumn);

    for (int rows = 0; rows < mat->row; rows++)
    {
        for (int cols = 0; cols < mat->collumn; cols++)
        {
            mat->data[rows][cols] = 0;
            for (int i = 0; i < mat1->collumn; i++)
            {
                mat->data[rows][cols] += mat1->data[rows][i] * mat2->data[i][cols];
            }
        }
    }

    return mat;
}

matrix_t *add_matrix(matrix_t *mat1, matrix_t *mat2)
{
    if (mat1->collumn != mat2->collumn || mat1->row != mat2->row)
    {
        printf("add matrix size error\n");
        fflush(stdout);
    }

    matrix_t *mat = create_matrix(mat1->row, mat1->collumn);

    for (int rows = 0; rows < mat->row; rows++)
    {
        for (int cols = 0; cols < mat->collumn; cols++)
        {
            mat->data[rows][cols] = mat1->data[rows][cols] + mat2->data[rows][cols];
        }
    }

    return mat;
}

matrix_t *subtract_matrix(matrix_t *mat1, matrix_t *mat2)
{
    if (mat1->collumn != mat2->collumn || mat1->row != mat2->row)
    {
        printf("subtract matrix size error\n");
        fflush(stdout);
    }

    matrix_t *mat = create_matrix(mat1->row, mat1->collumn);

    for (int rows = 0; rows < mat->row; rows++)
    {
        for (int cols = 0; cols < mat->collumn; cols++)
        {
            mat->data[rows][cols] = mat1->data[rows][cols] - mat2->data[rows][cols];
        }
    }

    return mat;
}

matrix_t *transpose_matrix(matrix_t *mat)
{
    matrix_t *ret_mat = create_matrix(mat->collumn, mat->row);

    for (int rows = 0; rows < ret_mat->row; rows++)
    {
        for (int cols = 0; cols < ret_mat->collumn; cols++)
        {
            ret_mat->data[rows][cols] = mat->data[cols][rows];
        }
    }

    return ret_mat;
}

matrix_t *import_img(char *addr)
{
    matrix_t *mat = create_matrix(30, 30);
    int width, height, bpp;

    uint8_t *rgb_image = stbi_load(addr, &width, &height, &bpp, 1);

    if (width * height != 900)
    {
        printf("input photo not!\n");
        exit(1);
    }

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            mat->data[i][j] = 255 - rgb_image[i * width + j];
        }
    }

    stbi_image_free(rgb_image);

    return mat;
}

matrix_t *import_img_1d(char *addr)
{
    matrix_t *mat = create_matrix(900, 1);
    int width, height, bpp;

    uint8_t *rgb_image = stbi_load(addr, &width, &height, &bpp, 1);

    if (width * height != 900)
    {
        printf("input photo not!\n");
        exit(1);
    }

    for (int i = 0; i < height * width; i++)
    {
        mat->data[i][0] = 255 - rgb_image[i];
    }

    stbi_image_free(rgb_image);

    return mat;
}

void normalize_matrix(matrix_t *mat, double in_min, double in_max, double out_min, double out_max)
{
    for (int rows = 0; rows < mat->row; rows++)
    {
        for (int cols = 0; cols < mat->collumn; cols++)
        {
            mat->data[rows][cols] = (mat->data[rows][cols] - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
        }
    }
}

void load_param(matrix_t *w1, matrix_t *b1, matrix_t *w2, matrix_t *b2, matrix_t *w3, matrix_t *b3)
{
    std::ifstream input("/home/surya/C/DSEC/nn/params/wnb4L.txt");
    char line_input[50];
    int counter = 0;
    int param = 0;

    for (std::string line; getline(input, line);)
    {
        // baca per line
        snprintf(line_input, 50, "%s", line.c_str());

        // line itu parameter apaan
        if (param == 0)
            w1->data[counter / input_layer_neuron][counter % input_layer_neuron] = atof(line_input);
        if (param == 1)
            b1->data[counter][0] = atof(line_input);
        if (param == 2)
            w2->data[counter / first_layer_neuron][counter % first_layer_neuron] = atof(line_input);
        if (param == 3)
            b2->data[counter][0] = atof(line_input);
        if (param == 4)
            w3->data[counter / second_layer_neuron][counter % second_layer_neuron] = atof(line_input);
        if (param == 5)
            b3->data[counter][0] = atof(line_input);

        counter++;
        if (param == 0 && counter == input_layer_neuron * first_layer_neuron)
        {
            param++;
            counter = 0;
        }

        if (param == 1 && counter == first_layer_neuron)
        {
            param++;
            counter = 0;
        }

        if (param == 2 && counter == first_layer_neuron * second_layer_neuron)
        {
            param++;
            counter = 0;
        }

        if (param == 3 && counter == second_layer_neuron)
        {
            param++;
            counter = 0;
        }

        if (param == 4 && counter == second_layer_neuron * output_layer_neuron)
        {
            param++;
            counter = 0;
        }

        if (param == 5 && counter == output_layer_neuron)
        {
            param++;
            counter = 0;
        }
        memset(line_input, 0, 50);
    }

    input.close();
}

void save_param(matrix_t *w1, matrix_t *b1, matrix_t *w2, matrix_t *b2, matrix_t *w3, matrix_t *b3)
{
    std::ofstream outfile("/home/surya/C/DSEC/nn/params/wnb4L.txt");

    char buf[50];
    memset(buf, 0, 50);

    for (int i = 0; i < input_layer_neuron * first_layer_neuron; i++)
    {
        outfile << w1->data[i / input_layer_neuron][i % input_layer_neuron] << "\t\tW1" << std::endl;
    }
    for (int i = 0; i < first_layer_neuron; i++)
    {
        outfile << b1->data[i][0] << "\t\tB1" << std::endl;
    }
    for (int i = 0; i < first_layer_neuron * second_layer_neuron; i++)
    {
        outfile << w2->data[i / first_layer_neuron][i % first_layer_neuron] << "\t\tW2" << std::endl;
    }
    for (int i = 0; i < second_layer_neuron; i++)
    {
        outfile << b2->data[i][0] << "\t\tB2" << std::endl;
    }
    for (int i = 0; i < second_layer_neuron * output_layer_neuron; i++)
    {
        outfile << w3->data[i / second_layer_neuron][i % second_layer_neuron] << "\t\tW3" << std::endl;
    }
    for (int i = 0; i < output_layer_neuron; i++)
    {
        outfile << b3->data[i][0] << "\t\tB3" << std::endl;
    }

    outfile.close();
}

void init_rand(matrix_t *w1, matrix_t *b1, matrix_t *w2, matrix_t *b2, matrix_t *w3, matrix_t *b3)
{
    std::ofstream outfile("/home/surya/C/DSEC/nn/params/wnb4L.txt");

    srand(time(NULL)); // Initialization, should only be called once.

    for (int i = 0; i < input_layer_neuron * first_layer_neuron; i++)
    {
        w1->data[i / input_layer_neuron][i % input_layer_neuron] = 2 * ((double)rand() / (double)RAND_MAX) - 1;
        outfile << w1->data[i / input_layer_neuron][i % input_layer_neuron] << "\t\tW1" << std::endl;
    }
    for (int i = 0; i < first_layer_neuron; i++)
    {
        // b1[i] = 2*((double)rand() / (double)RAND_MAX) - 1;
        b1->data[i][0] = 0;
        outfile << b1->data[i][0] << "\t\tB1" << std::endl;
    }
    for (int i = 0; i < first_layer_neuron * second_layer_neuron; i++)
    {
        w2->data[i / first_layer_neuron][i % first_layer_neuron] = 2 * ((double)rand() / (double)RAND_MAX) - 1;
        outfile << w2->data[i / first_layer_neuron][i % first_layer_neuron] << "\t\tW2" << std::endl;
    }
    for (int i = 0; i < second_layer_neuron; i++)
    {
        // b2[i] = 2*((double)rand() / (double)RAND_MAX) - 1;
        b2->data[i][0] = 0;
        outfile << b2->data[i][0] << "\t\tB2" << std::endl;
    }

    for (int i = 0; i < second_layer_neuron * output_layer_neuron; i++)
    {
        w3->data[i / second_layer_neuron][i % second_layer_neuron] = 2 * ((double)rand() / (double)RAND_MAX) - 1;
        outfile << w3->data[i / second_layer_neuron][i % second_layer_neuron] << "\t\tW3" << std::endl;
    }
    for (int i = 0; i < output_layer_neuron; i++)
    {
        // b2[i] = 2*((double)rand() / (double)RAND_MAX) - 1;
        b3->data[i][0] = 0;
        outfile << b3->data[i][0] << "\t\tB3" << std::endl;
    }

    outfile.close();
}

matrix_t *sigmoid_tf(matrix_t *mat)
{
    matrix_t *ret_mat = create_matrix(mat->row, mat->collumn);

    for (int rows = 0; rows < mat->row; rows++)
    {
        for (int cols = 0; cols < mat->collumn; cols++)
        {
            ret_mat->data[rows][cols] = 1.0 / (1.0 + exp(-1.0 * mat->data[rows][cols]));
        }
    }

    return ret_mat;
}

matrix_t *relu_tf(matrix_t *mat)
{
    matrix_t *ret_mat = create_matrix(mat->row, mat->collumn);

    for (int rows = 0; rows < mat->row; rows++)
    {
        for (int cols = 0; cols < mat->collumn; cols++)
        {
            ret_mat->data[rows][cols] = mat->data[rows][cols] >= 0 ? mat->data[rows][cols] : 0.0;
        }
    }

    return ret_mat;
}

matrix_t *sigmoid_dev(matrix_t *mat)
{
    matrix_t *ret_mat = create_matrix(mat->row, mat->collumn);
    double sigmoid = 0;
    for (int rows = 0; rows < mat->row; rows++)
    {
        for (int cols = 0; cols < mat->collumn; cols++)
        {
            sigmoid = 1.0 / (1.0 + exp(-1.0 * mat->data[rows][cols]));
            ret_mat->data[rows][cols] = sigmoid * (1.0 - sigmoid);
        }
    }

    return ret_mat;
}
matrix_t *relu_dev(matrix_t *mat)
{
    matrix_t *ret_mat = create_matrix(mat->row, mat->collumn);

    for (int rows = 0; rows < mat->row; rows++)
    {
        for (int cols = 0; cols < mat->collumn; cols++)
        {
            ret_mat->data[rows][cols] = mat->data[rows][cols] >= 0 ? 1.0 : 0.0;
        }
    }

    return ret_mat;
}

matrix_t *create_output(int correct_num)
{
    matrix_t *mat = create_matrix(output_layer_neuron, 1);

    for (int i = 0; i < output_layer_neuron; i++)
    {
        mat->data[i][0] = (i == correct_num);
    }

    return mat;
}

matrix_t *copy_matrix(matrix_t *mat)
{
    matrix_t *ret_mat = create_matrix(mat->row, mat->collumn);

    memcpy(ret_mat->data, mat->data, mat->collumn * mat->row * sizeof(double));

    return ret_mat;
}

void print_matrix_size(matrix_t *mat)
{
    printf("Size : %dx%d\n", mat->row, mat->collumn);
}

matrix_t *multiply_matrix_same_size(matrix_t *mat1, matrix_t *mat2)
{
    if (mat1->row != mat2->row || mat2->collumn != mat2->collumn)
    {
        printf("matrix multiply (same size) error\n");
    }

    matrix_t *mat = create_matrix(mat1->row, mat2->collumn);

    for (int rows = 0; rows < mat->row; rows++)
    {
        for (int cols = 0; cols < mat->collumn; cols++)
        {
            mat->data[rows][cols] = mat1->data[rows][cols] * mat2->data[rows][cols];
        }
    }

    return mat;
}

void gradient_descent(matrix_t *mat, matrix_t *mat_correction, double learning_rate)
{
    if (mat->row != mat_correction->row || mat->collumn != mat_correction->collumn)
    {
        printf("gradient descent matrix size error\n");
    }

    for (int rows = 0; rows < mat->row; rows++)
    {
        for (int cols = 0; cols < mat->collumn; cols++)
        {
            mat->data[rows][cols] -= mat_correction->data[rows][cols] * learning_rate;
        }
    }
}

matrix_t *gradient_descent_ret(matrix_t *mat, matrix_t *mat_correction, double learning_rate)
{
    if (mat->row != mat_correction->row || mat->collumn != mat_correction->collumn)
    {
        printf("gradient descent matrix size error\n");
    }

    matrix_t *ret_mat = create_matrix(mat->row, mat->collumn);

    for (int rows = 0; rows < mat->row; rows++)
    {
        for (int cols = 0; cols < mat->collumn; cols++)
        {
            ret_mat->data[rows][cols] = mat->data[rows][cols] - (mat_correction->data[rows][cols] * learning_rate);
        }
    }

    return ret_mat;
}

void multiply_matrix(matrix_t *mat1, matrix_t *mat2, matrix_t *ret_mat)
{
    if (mat1->collumn != mat2->row)
    {
        printf("multiply matrix size error\n");
        fflush(stdout);
    }
    if (mat1->row != ret_mat->row || mat2->collumn != ret_mat->collumn)
    {
        printf("multiply matrix ret mat size error\n");
        fflush(stdout);
    }

    for (int rows = 0; rows < ret_mat->row; rows++)
    {
        for (int cols = 0; cols < ret_mat->collumn; cols++)
        {
            ret_mat->data[rows][cols] = 0;
            for (int i = 0; i < mat1->collumn; i++)
            {
                ret_mat->data[rows][cols] += mat1->data[rows][i] * mat2->data[i][cols];
            }
        }
    }
}

void divide_matrix_scalar(matrix_t *mat, double divider)
{
    for (int rows = 0; rows < mat->row; rows++)
    {
        for (int cols = 0; cols < mat->collumn; cols++)
        {
            mat->data[rows][cols] = mat->data[rows][cols] / divider;
        }
    }
}

void add_matrix(matrix_t *mat1, matrix_t *mat2, matrix_t *ret_mat)
{
    if (mat1->collumn != mat2->collumn || mat1->row != mat2->row)
    {
        printf("add matrix size error\n");
        fflush(stdout);
    }

    // matrix_t* mat = create_matrix(mat1->row, mat1->collumn);
    if (ret_mat->row != mat1->row || ret_mat->collumn != mat1->collumn)
    {
        printf("add matrix ret mat size error\n");
        fflush(stdout);
    }

    for (int rows = 0; rows < ret_mat->row; rows++)
    {
        for (int cols = 0; cols < ret_mat->collumn; cols++)
        {
            ret_mat->data[rows][cols] = mat1->data[rows][cols] + mat2->data[rows][cols];
        }
    }
}

void subtract_matrix(matrix_t *mat1, matrix_t *mat2, matrix_t *ret_mat)
{
    if (mat1->collumn != mat2->collumn || mat1->row != mat2->row)
    {
        printf("subtract matrix size error\n");
        fflush(stdout);
    }

    // matrix_t* mat = create_matrix(mat1->row, mat1->collumn);
    if (ret_mat->row != mat1->row || ret_mat->collumn != mat1->collumn)
    {
        printf("subtract matrix ret mat size error\n");
        fflush(stdout);
    }

    for (int rows = 0; rows < ret_mat->row; rows++)
    {
        for (int cols = 0; cols < ret_mat->collumn; cols++)
        {
            ret_mat->data[rows][cols] = mat1->data[rows][cols] - mat2->data[rows][cols];
        }
    }
}

void transpose_matrix(matrix_t *mat, matrix_t *ret_mat)
{
    // matrix_t* ret_mat = create_matrix(mat->collumn, mat->row);
    if (ret_mat->row != mat->collumn || ret_mat->collumn != mat->row)
    {
        printf("transpose matrix wrong size");
        fflush(stdout);
    }

    for (int rows = 0; rows < ret_mat->row; rows++)
    {
        for (int cols = 0; cols < ret_mat->collumn; cols++)
        {
            ret_mat->data[rows][cols] = mat->data[cols][rows];
        }
    }
}

void import_img(char *addr, matrix_t *ret_mat)
{
    // matrix_t* mat = create_matrix(30,30);
    if (ret_mat->row != 30 || ret_mat->collumn != 30)
    {
        printf("import image wrong size\n");
        fflush(stdout);
    }
    int width, height, bpp;

    uint8_t *rgb_image = stbi_load(addr, &width, &height, &bpp, 1);

    if (width * height != 900)
    {
        printf("input photo not!\n");
        exit(1);
    }

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            ret_mat->data[i][j] = 255 - rgb_image[i * width + j];
        }
    }

    stbi_image_free(rgb_image);
}

void import_img_1d(char *addr, matrix_t *ret_mat)
{
    // matrix_t* mat = create_matrix(900,1);
    if (ret_mat->row != 900 || ret_mat->collumn != 1)
    {
        printf("import image 1d wrong size\n");
        fflush(stdout);
    }
    int width, height, bpp;

    uint8_t *rgb_image = stbi_load(addr, &width, &height, &bpp, 1);

    if (width * height != 900)
    {
        printf("input photo wring size!\n");
        exit(1);
    }

    for (int i = 0; i < height * width; i++)
    {
        ret_mat->data[i][0] = 255 - rgb_image[i];
    }

    stbi_image_free(rgb_image);
}

void sigmoid_tf(matrix_t *mat, matrix_t *ret_mat)
{
    // matrix_t* ret_mat = create_matrix(mat->row, mat->collumn);
    if (ret_mat->row != mat->row || ret_mat->collumn != mat->collumn)
    {
        printf("sigmoid tf ret mat size error\n");
        fflush(stdout);
    }

    for (int rows = 0; rows < mat->row; rows++)
    {
        for (int cols = 0; cols < mat->collumn; cols++)
        {
            ret_mat->data[rows][cols] = 1.0 / (1.0 + exp(-1.0 * mat->data[rows][cols]));
        }
    }

    // return ret_mat;
}

void relu_tf(matrix_t *mat, matrix_t *ret_mat)
{
    // matrix_t* ret_mat = create_matrix(mat->row, mat->collumn);
    if (ret_mat->row != mat->row || ret_mat->collumn != mat->collumn)
    {
        printf("relu tf ret mat size error\n");
        fflush(stdout);
    }

    for (int rows = 0; rows < mat->row; rows++)
    {
        for (int cols = 0; cols < mat->collumn; cols++)
        {
            ret_mat->data[rows][cols] = mat->data[rows][cols] >= 0 ? mat->data[rows][cols] : 0.0;
        }
    }

    // return ret_mat;
}

void sigmoid_dev(matrix_t *mat, matrix_t *ret_mat)
{
    // matrix_t* ret_mat = create_matrix(mat->row, mat->collumn);

    if (ret_mat->row != mat->row || ret_mat->collumn != mat->collumn)
    {
        printf("sigmoid dev ret mat size error\n");
        fflush(stdout);
    }

    double sigmoid = 0;
    for (int rows = 0; rows < mat->row; rows++)
    {
        for (int cols = 0; cols < mat->collumn; cols++)
        {
            sigmoid = 1.0 / (1.0 + exp(-1.0 * mat->data[rows][cols]));
            ret_mat->data[rows][cols] = sigmoid * (1.0 - sigmoid);
        }
    }

    // return ret_mat;
}
void relu_dev(matrix_t *mat, matrix_t *ret_mat)
{
    // matrix_t* ret_mat = create_matrix(mat->row, mat->collumn);
    if (ret_mat->row != mat->row || ret_mat->collumn != mat->collumn)
    {
        printf("relu dev ret mat size error\n");
        fflush(stdout);
    }

    for (int rows = 0; rows < mat->row; rows++)
    {
        for (int cols = 0; cols < mat->collumn; cols++)
        {
            ret_mat->data[rows][cols] = mat->data[rows][cols] >= 0 ? 1.0 : 0.0;
        }
    }

    // return ret_mat;
}

void create_output(int correct_num, matrix_t *ret_mat)
{
    // matrix_t* mat = create_matrix(output_layer_neuron,1);
    if (ret_mat->row != output_layer_neuron || ret_mat->collumn != 1)
    {
        printf("create output mat size error\n");
        fflush(stdout);
    }

    for (int i = 0; i < output_layer_neuron; i++)
    {
        ret_mat->data[i][0] = (i == correct_num ? 1.0 : 0.0);
    }

    // return mat;
}

void copy_matrix(matrix_t *mat, matrix_t *ret_mat)
{
    // matrix_t* ret_mat = create_matrix(mat->row, mat->collumn);
    if (ret_mat->row != mat->row || ret_mat->collumn != mat->collumn)
    {
        printf("copy mat size error\n");
        fflush(stdout);
    }

    memcpy(ret_mat->data, mat->data, mat->collumn * mat->row * sizeof(double));

    // return ret_mat;
}

void multiply_matrix_same_size(matrix_t *mat1, matrix_t *mat2, matrix_t *ret_mat)
{
    if (mat1->row != mat2->row || mat2->collumn != mat2->collumn)
    {
        printf("matrix multiply (same size) error\n");
        fflush(stdout);
    }

    if (ret_mat->row != mat1->row || ret_mat->collumn != mat1->collumn)
    {
        printf("relu tf ret mat size error\n");
        fflush(stdout);
    }

    // matrix_t* mat = create_matrix(mat1->row, mat2->collumn);

    for (int rows = 0; rows < ret_mat->row; rows++)
    {
        for (int cols = 0; cols < ret_mat->collumn; cols++)
        {
            ret_mat->data[rows][cols] = mat1->data[rows][cols] * mat2->data[rows][cols];
        }
    }

    // return mat;
}

void gradient_descent(matrix_t *mat, matrix_t *mat_correction, double learning_rate, matrix_t *ret_mat)
{
    if (mat->row != mat_correction->row || mat->collumn != mat_correction->collumn)
    {
        printf("gradient descent matrix size error\n");
    }

    // matrix_t* ret_mat = create_matrix(mat->row, mat->collumn);
    if (ret_mat->row != mat->row || ret_mat->collumn != mat->collumn)
    {
        printf("gradien descent ret mat size error\n");
        fflush(stdout);
    }

    for (int rows = 0; rows < mat->row; rows++)
    {
        for (int cols = 0; cols < mat->collumn; cols++)
        {
            ret_mat->data[rows][cols] = mat->data[rows][cols] - (mat_correction->data[rows][cols] * learning_rate);
        }
    }

    // return ret_mat;
}

double highest_output(matrix_t *mat)
{
    if (mat->row != output_layer_neuron || mat->collumn != 1)
    {
        printf("create output mat size error\n");
        fflush(stdout);
    }

    int highest_num = 0;
    double highest_value = 0;

    for (int i = 0; i < output_layer_neuron; i++)
    {
        if (mat->data[i][0] > highest_value)
        {
            highest_num = i;
            highest_value = mat->data[i][0];
        }
    }

    return highest_num + highest_value;
}

void print_output(matrix_t *mat)
{
    char letters[5] = {'D', 'A', 'N', 'E', 'R'};
    for (int rows = 0; rows < mat->row; rows++)
    {
        for (int cols = 0; cols < mat->collumn; cols++)
        {
            if (mat->data[rows][cols] >= 0)
                printf("%c: %.4f  ", letters[rows], mat->data[rows][cols]);
            else
                printf("%c: %.3f  ", letters[rows], mat->data[rows][cols]);
        }
        printf("\n");
    }
}