import java.io.*;
import java.util.*;

public class PolynomialRegression {

    static double[][] transpose(double[][] a) {
        double[][] t = new double[a[0].length][a.length];
        for (int i = 0; i < a.length; i++)
            for (int j = 0; j < a[0].length; j++)
                t[j][i] = a[i][j];
        return t;
    }

    static double[][] multiply(double[][] a, double[][] b) {
        double[][] c = new double[a.length][b[0].length];
        for (int i = 0; i < a.length; i++)
            for (int j = 0; j < b[0].length; j++)
                for (int k = 0; k < b.length; k++)
                    c[i][j] += a[i][k] * b[k][j];
        return c;
    }

    static double[][] inverse(double[][] a) {
        int n = a.length;
        double[][] aug = new double[n][2 * n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
                aug[i][j] = a[i][j];
            aug[i][i + n] = 1;
        }

        for (int i = 0; i < n; i++) {
            int max = i;

            for (int j = i + 1; j < n; j++)
                if (Math.abs(aug[j][i]) > Math.abs(aug[max][i]))
                    max = j;

            double[] temp = aug[i];
            aug[i] = aug[max];
            aug[max] = temp;

            double pivot = aug[i][i];

            for (int j = 0; j < 2 * n; j++)
                aug[i][j] /= pivot;

            for (int j = 0; j < n; j++) {
                if (j == i) continue;

                double factor = aug[j][i];

                for (int k = 0; k < 2 * n; k++)
                    aug[j][k] -= factor * aug[i][k];
            }
        }

        double[][] inv = new double[n][n];

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                inv[i][j] = aug[i][j + n];

        return inv;
    }

    static double[] fit(double[] x, double[] y, int degree) {
        double[][] X = new double[x.length][degree + 1];

        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j <= degree; j++)
                X[i][j] = Math.pow(x[i], j);
        }

        double[][] Xt = transpose(X);
        double[][] XtX = multiply(Xt, X);
        double[][] inv = inverse(XtX);

        double[][] Y = new double[y.length][1];

        for (int i = 0; i < y.length; i++)
            Y[i][0] = y[i];

        double[][] coefficients = multiply(multiply(inv, Xt), Y);
        double[] result = new double[degree + 1];

        for (int i = 0; i <= degree; i++)
            result[i] = coefficients[i][0];

        return result;
    }

    static double predict(double x, double[] coefficients) {
        double y = 0;

        for (int i = 0; i < coefficients.length; i++)
            y += coefficients[i] * Math.pow(x, i);

        return y;
    }

    static double mse(double[] x, double[] y, double[] coefficients) {
        double error = 0;

        for (int i = 0; i < x.length; i++) {
            double difference = y[i] - predict(x[i], coefficients);
            error += difference * difference;
        }

        return error / x.length;
    }

    static double[][] split(double[][] data, int start, int end) {
        double[][] result = new double[end - start][2];

        for (int i = start; i < end; i++) {
            result[i - start][0] = data[i][0];
            result[i - start][1] = data[i][1];
        }

        return result;
    }

    public static void main(String[] args) throws Exception {

        String file = "noisy_10.txt";

        ArrayList<double[]> list = new ArrayList<>();

        Scanner sc = new Scanner(new File(file));

        while (sc.hasNextDouble()) {
            double x = sc.nextDouble();
            double y = sc.nextDouble();
            list.add(new double[]{x, y});
        }

        sc.close();

        double[][] data = new double[list.size()][2];

        for (int i = 0; i < list.size(); i++) data[i] = list.get(i);

        Random random = new Random(42);

        for (int i = data.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            double[] temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }

        int n = data.length;

        int trainSize = (int)(0.6 * n);
        int testSize = (int)(0.2 * n);

        double[][] train = split(data, 0, trainSize);
        double[][] test = split(data, trainSize, trainSize + testSize);
        double[][] validation = split(data, trainSize + testSize, n);

        double[] trainX = new double[train.length];
        double[] trainY = new double[train.length];

        double[] testX = new double[test.length];
        double[] testY = new double[test.length];

        double[] valX = new double[validation.length];
        double[] valY = new double[validation.length];

        for (int i = 0; i < train.length; i++) {
            trainX[i] = train[i][0];
            trainY[i] = train[i][1];
        }

        for (int i = 0; i < test.length; i++) {
            testX[i] = test[i][0];
            testY[i] = test[i][1];
        }

        for (int i = 0; i < validation.length; i++) {
            valX[i] = validation[i][0];
            valY[i] = validation[i][1];
        }

        int bestDegree = 1;
        double bestTestError = Double.MAX_VALUE;
        double[] bestCoefficients = null;

        for (int degree = 1; degree <= 10; degree++) {

            double[] coefficients = fit(trainX, trainY, degree);

            double trainError = mse(trainX, trainY, coefficients);
            double testError = mse(testX, testY, coefficients);

            System.out.println("Degree " + degree +" | Train MSE = " + trainError +" | Test MSE = " + testError);
            
            if (testError < bestTestError) {
                bestTestError = testError;
                bestDegree = degree;
                bestCoefficients = coefficients;
            }
        }

        double validationError = mse(valX, valY, bestCoefficients);

        System.out.println("\nBest Degree: " + bestDegree);
        System.out.println("Best Test MSE: " + bestTestError);
        System.out.println("Validation MSE: " + validationError);

        System.out.println("\nPolynomial:");

        for (int i = 0; i < bestCoefficients.length; i++) {
            System.out.printf("%+.16fx^%d ", bestCoefficients[i], i);
        }

        System.out.println();
    }
}