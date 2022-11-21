#include "neural_network/3L/main.h"
#include "neural_network/matplotlibcpp.h"

namespace plt = matplotlibcpp;

int main()
{
    /*
        matrix_t* mat1 = create_matrix(10,2);

        mat1->data[1][0] = 2.8;
        mat1->data[8][0] = 10;
        mat1->data[9][0] = 8.2;

        matrix_t* mat2 = create_matrix(10,2);

        mat2->data[1][0] = 3.1;
        mat2->data[8][0] = 9.2;

        fill_matrix(mat1, 1);
        fill_matrix(mat2, -9);

        add_matrix(mat1, mat2, mat1);

        print_matrix(mat1);

        return 0;
    */
    char in_buf[60];
    snprintf(in_buf, 60, "/home/dancoeks/Kuliah/DSEC/NN/Training set/D/D2.jpg");

    matrix_t *input_img = create_matrix(900, 1);
    input_img = import_img_1d(in_buf);
    // print_matrix(input_img);

    normalize_matrix(input_img, 0, 255, 0.0, 1.0);
    // print_matrix(input_img);

    matrix_t *w1 = create_matrix(first_layer_neuron, input_layer_neuron);
    matrix_t *b1 = create_matrix(first_layer_neuron, 1);

    matrix_t *w2 = create_matrix(output_layer_neuron, first_layer_neuron);
    matrix_t *b2 = create_matrix(output_layer_neuron, 1);

    init_rand(w1, b1, w2, b2);
    // load_param(w1, b1, w2, b2);

    matrix_t *wp1 = create_matrix(first_layer_neuron, 1);
    matrix_t *z1 = create_matrix(first_layer_neuron, 1);
    matrix_t *a1 = create_matrix(first_layer_neuron, 1);

    matrix_t *wp2 = create_matrix(output_layer_neuron, 1);
    matrix_t *z2 = create_matrix(output_layer_neuron, 1);
    matrix_t *a2 = create_matrix(output_layer_neuron, 1);
    matrix_t *y = create_matrix(output_layer_neuron, 1);

    multiply_matrix(w1, input_img, wp1);
    add_matrix(wp1, b1, z1);
    sigmoid_tf(z1, a1);

    multiply_matrix(w2, a1, wp2);
    add_matrix(wp2, b2, z2);
    sigmoid_tf(z2, a2);
    print_matrix(a2);
    printf("\n");

    matrix_t *dz2 = create_matrix(output_layer_neuron, 1);
    matrix_t *dw2 = create_matrix(output_layer_neuron, first_layer_neuron);
    matrix_t *db2 = create_matrix(output_layer_neuron, 1);

    matrix_t *g_dev = create_matrix(first_layer_neuron, 1);
    matrix_t *dz1 = create_matrix(first_layer_neuron, 1);
    matrix_t *dw1 = create_matrix(first_layer_neuron, input_layer_neuron);
    matrix_t *db1 = create_matrix(first_layer_neuron, 1);

    matrix_t *all_dz2 = create_matrix(output_layer_neuron, 1);
    matrix_t *all_dw2 = create_matrix(output_layer_neuron, first_layer_neuron);
    matrix_t *all_db2 = create_matrix(output_layer_neuron, 1);

    matrix_t *all_dz1 = create_matrix(first_layer_neuron, 1);
    matrix_t *all_dw1 = create_matrix(first_layer_neuron, input_layer_neuron);
    matrix_t *all_db1 = create_matrix(first_layer_neuron, 1);

    matrix_t *a1_T = create_matrix(1, first_layer_neuron);
    matrix_t *w2_T = create_matrix(first_layer_neuron, output_layer_neuron);
    matrix_t *input_img_T = create_matrix(1, 900);

    char letters[5] = {'D', 'A', 'N', 'E', 'R'};

    for (int epoch = 0; epoch < used_epoch; epoch++)
    {
        printf("Epoch: %d\r", epoch + 1);
        fflush(stdout);
        for (int num_training = 1; num_training <= 10; num_training++)
        {
            for (int letter = 0; letter < 5; letter++)
            {
                snprintf(in_buf, 60, "/home/dancoeks/Kuliah/DSEC/NN/Training set/%c/%c%d.jpg", letters[letter], letters[letter], num_training);
                import_img_1d(in_buf, input_img);

                normalize_matrix(input_img, 0.0, 255.0, 0.0, 1.0);

                // forward propagation
                multiply_matrix(w1, input_img, wp1);
                add_matrix(wp1, b1, z1);
                sigmoid_tf(z1, a1);

                multiply_matrix(w2, a1, wp2);
                add_matrix(wp2, b2, z2);
                sigmoid_tf(z2, a2);

                // backward propagation
                create_output(letter, y);
                subtract_matrix(a2, y, dz2);
                transpose_matrix(a1, a1_T);
                multiply_matrix(dz2, a1_T, dw2);
                copy_matrix(dz2, db2);

                transpose_matrix(w2, w2_T);
                multiply_matrix(w2_T, dz2, dz1);
                sigmoid_dev(z1, g_dev);
                multiply_matrix_same_size(dz1, g_dev, dz1);
                transpose_matrix(input_img, input_img_T);
                multiply_matrix(dz1, input_img_T, dw1);
                copy_matrix(dz1, db1);

                add_matrix(all_dw2, dw2, all_dw2);
                add_matrix(all_db2, db2, all_db2);
                add_matrix(all_dw1, dw1, all_dw1);
                add_matrix(all_db1, db1, all_db1);
            }

            divide_matrix_scalar(all_db2, 5);
            divide_matrix_scalar(all_db1, 5);

            gradient_descent(w2, all_dw2, 0.01);
            gradient_descent(b2, all_db2, 0.01);
            gradient_descent(w1, all_dw1, 0.01);
            gradient_descent(b1, all_db1, 0.01);

            fill_matrix(all_dw2, 0.0);
            fill_matrix(all_db2, 0.0);
            fill_matrix(all_dw1, 0.0);
            fill_matrix(all_db1, 0.0);

            // multiply_matrix(w1, input_img, wp1);
            // add_matrix(wp1, b1, z1);
            // sigmoid_tf(z1, a1);

            // multiply_matrix(w2, a1, wp2);
            // add_matrix(wp2, b2, z2);
            // sigmoid_tf(z2, a2);
        }
        int correct_guess = 0;
        int num_guess = 0;
        for (int num_training = 11; num_training <= 15; num_training++)
        {
            for (int letter = 0; letter < 5; letter++)
            {
                snprintf(in_buf, 60, "/home/dancoeks/Kuliah/DSEC/NN/Training set/%c/%c%d.jpg", letters[letter], letters[letter], num_training);
                import_img_1d(in_buf, input_img);

                normalize_matrix(input_img, 0.0, 255.0, 0.0, 1.0);

                // forward propagation
                multiply_matrix(w1, input_img, wp1);
                add_matrix(wp1, b1, z1);
                sigmoid_tf(z1, a1);

                multiply_matrix(w2, a1, wp2);
                add_matrix(wp2, b2, z2);
                sigmoid_tf(z2, a2);

                double output = highest_output(a2);
                if ((int)output == letter)
                    correct_guess++;
                num_guess++;
                efficiency.at(epoch) = (double)correct_guess / (double)num_guess;
            }
        }
        correct_guesses.at(epoch) = correct_guess;
        epochs.at(epoch) = epoch;
    }

    plt::plot(epochs, correct_guesses);
    plt::xlabel("Epoch");
    plt::ylabel("Correct Guesses");
    plt::show();

    plt::plot(epochs, efficiency);
    plt::xlabel("Epoch");
    plt::ylabel("Efficiency");
    plt::show();
    // dz2 = subtract_matrix(a2, create_output(1));
    // dw2 = multiply_matrix(dz2, transpose_matrix(a1));
    // db2 = copy_matrix(dz2);

    // dz1 = multiply_matrix(transpose_matrix(w2), dz2);
    // dw1 = multiply_matrix(dz1, transpose_matrix(input_img));
    // db1 = copy_matrix(dz1);

    // gradient_descent(w2, all_dw2, 0.1);
    // gradient_descent(b2, all_db2, 0.1);
    // gradient_descent(w1, all_dw1, 0.1);
    // gradient_descent(b1, all_db1, 0.1);

    int correct_guess = 0;
    int num_guess = 0;
    for (int num_training = 11; num_training <= 15; num_training++)
    {
        for (int letter = 0; letter < 5; letter++)
        {
            snprintf(in_buf, 60, "/home/dancoeks/Kuliah/DSEC/NN/Training set/%c/%c%d.jpg", letters[letter], letters[letter], num_training);
            import_img_1d(in_buf, input_img);

            normalize_matrix(input_img, 0.0, 255.0, 0.0, 1.0);

            // forward propagation
            multiply_matrix(w1, input_img, wp1);
            add_matrix(wp1, b1, z1);
            sigmoid_tf(z1, a1);

            multiply_matrix(w2, a1, wp2);
            add_matrix(wp2, b2, z2);
            sigmoid_tf(z2, a2);

            double output = highest_output(a2);
            if ((int)output == letter)
                correct_guess++;
            num_guess++;
        }
    }

    printf("Correct guess:%d out of %d\n", correct_guess, num_guess);
    // save_param(w1, b1, w2, b2);

    // while (1)
    // {
    //     char input[5];
    //     scanf("%s", input);
    //     for (int i = 0; i < 6; i++)
    //     {
    //         if (i == 5)
    //         {
    //             input[0] = 0;
    //             break;
    //         }
    //         if (input[0] == letters[i])
    //             break;
    //     }
    //     if (input[0] == 0)
    //     {
    //         printf("wrong letter input\n");
    //         continue;
    //     }

    //     int num = atoi(input + 1);
    //     if (num < 1 || num > 15)
    //     {
    //         printf("wrong num");
    //         continue;
    //     }
    //     char letter = input[0];

    //     snprintf(in_buf, 60, "/home/dancoeks/Kuliah/DSEC/NN/Training set/%c/%c%d.jpg", letter, letter, num);
    //     import_img_1d(in_buf, input_img);

    //     normalize_matrix(input_img, 0.0, 255.0, 0.0, 1.0);

    //     multiply_matrix(w1, input_img, wp1);
    //     add_matrix(wp1, b1, z1);
    //     sigmoid_tf(z1, a1);

    //     multiply_matrix(w2, a1, wp2);
    //     add_matrix(wp2, b2, z2);
    //     sigmoid_tf(z2, a2);

    //     printf("\n");
    //     print_output(a2);
    //     printf("\n");
    // }
    // return 0;
}
