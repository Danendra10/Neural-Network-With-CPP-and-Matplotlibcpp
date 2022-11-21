#include "neural_network/5L/main.h"
#include "neural_network/matplotlibcpp.h"

namespace plt = matplotlibcpp;

int main()
{
    char in_buf[60];
    snprintf(in_buf, 60, "/home/dancoeks/Kuliah/DSEC/NN/Training set/D/D2.jpg");

    matrix_t *input_img = create_matrix(900, 1);
    input_img = import_img_1d(in_buf);

    normalize_matrix(input_img, 0, 255, 0.0, 1.0);
    // print_matrix(input_img);

    matrix_t *w1 = create_matrix(first_layer_neuron, input_layer_neuron);
    matrix_t *b1 = create_matrix(first_layer_neuron, 1);

    matrix_t *w2 = create_matrix(second_layer_neuron, first_layer_neuron);
    matrix_t *b2 = create_matrix(second_layer_neuron, 1);

    matrix_t *w3 = create_matrix(third_layer_neuron, second_layer_neuron);
    matrix_t *b3 = create_matrix(third_layer_neuron, 1);

    matrix_t *w4 = create_matrix(output_layer_neuron, third_layer_neuron);
    matrix_t *b4 = create_matrix(output_layer_neuron, 1);

    init_rand(w1, b1, w2, b2, w3, b3, w4, b4);
    // load_param(w1, b1, w2, b2, w3, b3, w4, b4);

    matrix_t *wp1 = create_matrix(first_layer_neuron, 1);
    matrix_t *z1 = create_matrix(first_layer_neuron, 1);
    matrix_t *a1 = create_matrix(first_layer_neuron, 1);

    matrix_t *wp2 = create_matrix(second_layer_neuron, 1);
    matrix_t *z2 = create_matrix(second_layer_neuron, 1);
    matrix_t *a2 = create_matrix(second_layer_neuron, 1);

    matrix_t *wp3 = create_matrix(third_layer_neuron, 1);
    matrix_t *z3 = create_matrix(third_layer_neuron, 1);
    matrix_t *a3 = create_matrix(third_layer_neuron, 1);

    matrix_t *wp4 = create_matrix(output_layer_neuron, 1);
    matrix_t *z4 = create_matrix(output_layer_neuron, 1);
    matrix_t *a4 = create_matrix(output_layer_neuron, 1);

    matrix_t *y = create_matrix(output_layer_neuron, 1);

    multiply_matrix(w1, input_img, wp1);
    add_matrix(wp1, b1, z1);
    // sigmoid_tf(z1, a1);
    sigmoid_tf(z1, a1);

    multiply_matrix(w2, a1, wp2);
    add_matrix(wp2, b2, z2);
    // sigmoid_tf(z2, a2);
    sigmoid_tf(z2, a2);

    multiply_matrix(w3, a2, wp3);
    add_matrix(wp3, b3, z3);
    sigmoid_tf(z3, a3);

    multiply_matrix(w4, a3, wp4);
    add_matrix(wp4, b4, z4);
    sigmoid_tf(z4, a4);

    print_matrix(a4);
    printf("\n");

    // return 0;
    matrix_t *dz4 = create_matrix(output_layer_neuron, 1);
    matrix_t *dw4 = create_matrix(output_layer_neuron, third_layer_neuron);
    matrix_t *db4 = create_matrix(output_layer_neuron, 1);

    matrix_t *dz3 = create_matrix(third_layer_neuron, 1);
    matrix_t *dw3 = create_matrix(third_layer_neuron, second_layer_neuron);
    matrix_t *db3 = create_matrix(third_layer_neuron, 1);

    matrix_t *dz2 = create_matrix(second_layer_neuron, 1);
    matrix_t *dw2 = create_matrix(second_layer_neuron, first_layer_neuron);
    matrix_t *db2 = create_matrix(second_layer_neuron, 1);

    matrix_t *dz1 = create_matrix(first_layer_neuron, 1);
    matrix_t *dw1 = create_matrix(first_layer_neuron, input_layer_neuron);
    matrix_t *db1 = create_matrix(first_layer_neuron, 1);

    matrix_t *all_dz4 = create_matrix(output_layer_neuron, 1);
    matrix_t *all_dw4 = create_matrix(output_layer_neuron, third_layer_neuron);
    matrix_t *all_db4 = create_matrix(output_layer_neuron, 1);

    matrix_t *all_dz3 = create_matrix(third_layer_neuron, 1);
    matrix_t *all_dw3 = create_matrix(third_layer_neuron, second_layer_neuron);
    matrix_t *all_db3 = create_matrix(third_layer_neuron, 1);

    matrix_t *all_dz2 = create_matrix(second_layer_neuron, 1);
    matrix_t *all_dw2 = create_matrix(second_layer_neuron, first_layer_neuron);
    matrix_t *all_db2 = create_matrix(second_layer_neuron, 1);

    matrix_t *all_dz1 = create_matrix(first_layer_neuron, 1);
    matrix_t *all_dw1 = create_matrix(first_layer_neuron, input_layer_neuron);
    matrix_t *all_db1 = create_matrix(first_layer_neuron, 1);

    matrix_t *g3_dev = create_matrix(third_layer_neuron, 1);
    matrix_t *g2_dev = create_matrix(second_layer_neuron, 1);
    matrix_t *g1_dev = create_matrix(first_layer_neuron, 1);

    matrix_t *a3_T = create_matrix(1, third_layer_neuron);
    matrix_t *a2_T = create_matrix(1, second_layer_neuron);
    matrix_t *a1_T = create_matrix(1, first_layer_neuron);

    matrix_t *w4_T = create_matrix(third_layer_neuron, output_layer_neuron);
    matrix_t *w3_T = create_matrix(second_layer_neuron, third_layer_neuron);
    matrix_t *w2_T = create_matrix(first_layer_neuron, second_layer_neuron);
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
                // sigmoid_tf(z1, a1);
                sigmoid_tf(z1, a1);

                multiply_matrix(w2, a1, wp2);
                add_matrix(wp2, b2, z2);
                // sigmoid_tf(z2, a2);
                sigmoid_tf(z2, a2);

                multiply_matrix(w3, a2, wp3);
                add_matrix(wp3, b3, z3);
                sigmoid_tf(z3, a3);

                multiply_matrix(w4, a3, wp4);
                add_matrix(wp4, b4, z4);
                sigmoid_tf(z4, a4);

                // backward propagation
                create_output(letter, y);
                subtract_matrix(a4, y, dz4);
                transpose_matrix(a3, a3_T);
                multiply_matrix(dz4, a3_T, dw4);
                copy_matrix(dz4, db4);

                transpose_matrix(w4, w4_T);
                multiply_matrix(w4_T, dz4, dz3);
                sigmoid_dev(z3, g3_dev);
                multiply_matrix_same_size(dz3, g3_dev, dz3);
                transpose_matrix(a2, a2_T);
                multiply_matrix(dz3, a2_T, dw3);
                copy_matrix(dz3, db3);

                transpose_matrix(w3, w3_T);
                multiply_matrix(w3_T, dz3, dz2);
                sigmoid_dev(z2, g2_dev);
                multiply_matrix_same_size(dz2, g2_dev, dz2);
                transpose_matrix(a1, a1_T);
                multiply_matrix(dz2, a1_T, dw2);
                copy_matrix(dz2, db2);

                transpose_matrix(w2, w2_T);
                multiply_matrix(w2_T, dz2, dz1);
                sigmoid_dev(z1, g1_dev);
                multiply_matrix_same_size(dz1, g1_dev, dz1);
                transpose_matrix(input_img, input_img_T);
                multiply_matrix(dz1, input_img_T, dw1);
                copy_matrix(dz1, db1);

                add_matrix(all_dw4, dw4, all_dw4);
                add_matrix(all_db4, db4, all_db4);
                add_matrix(all_dw3, dw3, all_dw3);
                add_matrix(all_db3, db3, all_db3);
                add_matrix(all_dw2, dw2, all_dw2);
                add_matrix(all_db2, db2, all_db2);
                add_matrix(all_dw1, dw1, all_dw1);
                add_matrix(all_db1, db1, all_db1);
            }

            divide_matrix_scalar(all_db4, 5);
            divide_matrix_scalar(all_db3, 5);
            divide_matrix_scalar(all_db2, 5);
            divide_matrix_scalar(all_db1, 5);

            gradient_descent(w4, all_dw4, 0.01);
            gradient_descent(b4, all_db4, 0.01);
            gradient_descent(w3, all_dw3, 0.01);
            gradient_descent(b3, all_db3, 0.01);
            gradient_descent(w2, all_dw2, 0.01);
            gradient_descent(b2, all_db2, 0.01);
            gradient_descent(w1, all_dw1, 0.01);
            gradient_descent(b1, all_db1, 0.01);

            fill_matrix(all_dw4, 0.0);
            fill_matrix(all_db4, 0.0);
            fill_matrix(all_dw3, 0.0);
            fill_matrix(all_db3, 0.0);
            fill_matrix(all_dw2, 0.0);
            fill_matrix(all_db2, 0.0);
            fill_matrix(all_dw1, 0.0);
            fill_matrix(all_db1, 0.0);
        }
        for (int training = 11; training <= 15; training++)
        {
            for (int letterr = 0; letterr < 5; letterr++)
            {
                snprintf(in_buf, 60, "/home/dancoeks/Kuliah/DSEC/NN/Training set/%c/%c%d.jpg", letters[letterr], letters[letterr], training);
                import_img_1d(in_buf, input_img);

                normalize_matrix(input_img, 0.0, 255.0, 0.0, 1.0);

                // forward propagation
                multiply_matrix(w1, input_img, wp1);
                add_matrix(wp1, b1, z1);
                // sigmoid_tf(z1, a1);
                sigmoid_tf(z1, a1);

                multiply_matrix(w2, a1, wp2);
                add_matrix(wp2, b2, z2);
                // sigmoid_tf(z2, a2);
                sigmoid_tf(z2, a2);

                multiply_matrix(w3, a2, wp3);
                add_matrix(wp3, b3, z3);
                sigmoid_tf(z3, a3);

                multiply_matrix(w4, a3, wp4);
                add_matrix(wp4, b4, z4);
                sigmoid_tf(z4, a4);

                double output = highest_output(a4);
                if ((int)output == letterr)
                    correct_guess++;
                num_guess++;
                efficiency.at(epoch) = (double)correct_guess / (double)num_guess;
            }
        }

        correct_guesses.at(epoch) = correct_guess;
        epochs.at(epoch) = epoch;
        // plot epoch as x and correct guess as y
        for (int i = 0; i < epoch; i++)
        {
            printf("%d %d\n", i, correct_guesses.at(i));
        }
        correct_guess = 0;
    }
    // save epochs and correct_guesses to file
    FILE *fp = fopen("correct_guesses.txt", "w");
    for (int i = 0; i < epochs.size(); i++)
    {
        fprintf(fp, "%d %d \n", epochs.at(i), correct_guesses.at(i));
    }
    fclose(fp);

    // make a plot out of file correct_guesses.txt with matplotlibcpp
    // give label
    plt::plot(epochs, correct_guesses);
    plt::xlabel("Epoch");
    plt::ylabel("Correct Guesses");
    plt::show();

    plt::plot(epochs, efficiency);
    plt::xlabel("Epoch");
    plt::ylabel("Efficiency");
    plt::show();

    snprintf(in_buf, 60, "/home/dancoeks/Kuliah/DSEC/NN/Training set/D/D15.jpg");
    import_img_1d(in_buf, input_img);

    normalize_matrix(input_img, 0.0, 255.0, 0.0, 1.0);

    multiply_matrix(w1, input_img, wp1);
    add_matrix(wp1, b1, z1);
    // sigmoid_tf(z1, a1);
    sigmoid_tf(z1, a1);

    multiply_matrix(w2, a1, wp2);
    add_matrix(wp2, b2, z2);
    // sigmoid_tf(z2, a2);
    sigmoid_tf(z2, a2);

    multiply_matrix(w3, a2, wp3);
    add_matrix(wp3, b3, z3);
    sigmoid_tf(z3, a3);

    multiply_matrix(w4, a3, wp4);
    add_matrix(wp4, b4, z4);
    sigmoid_tf(z4, a4);

    printf("\n");
    print_matrix(a4);
    printf("\n");

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
            // sigmoid_tf(z1, a1);
            sigmoid_tf(z1, a1);

            multiply_matrix(w2, a1, wp2);
            add_matrix(wp2, b2, z2);
            // sigmoid_tf(z2, a2);
            sigmoid_tf(z2, a2);

            multiply_matrix(w3, a2, wp3);
            add_matrix(wp3, b3, z3);
            sigmoid_tf(z3, a3);

            multiply_matrix(w4, a3, wp4);
            add_matrix(wp4, b4, z4);
            sigmoid_tf(z4, a4);

            double output = highest_output(a4);
            if ((int)output == letter)
                correct_guess++;
            num_guess++;
        }
    }

    printf("Correct guess:%d out of %d\n", correct_guess, num_guess);
    // save_param(w1, b1, w2, b2, w3, b3, w4, b4);

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
    //     // sigmoid_tf(z1, a1);
    //     sigmoid_tf(z1, a1);

    //     multiply_matrix(w2, a1, wp2);
    //     add_matrix(wp2, b2, z2);
    //     // sigmoid_tf(z2, a2);
    //     sigmoid_tf(z2, a2);

    //     multiply_matrix(w3, a2, wp3);
    //     add_matrix(wp3, b3, z3);
    //     sigmoid_tf(z3, a3);

    //     multiply_matrix(w4, a3, wp4);
    //     add_matrix(wp4, b4, z4);
    //     sigmoid_tf(z4, a4);

    //     printf("\n");
    //     print_output(a4);
    //     printf("\n");
    // }

    // plt::plot({1, 3, 2, 4});
    // plt::show();
}
