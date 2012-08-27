
import java.io.*;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Scanner;

/**
 *
 * @author Agathoklis Georgiou
 */
public class Solution {

    static ArrayList<exampleObj> exampleList = new ArrayList<exampleObj>();
    static double inputWeights[][];
    static double hiddenWeights[];
    static double RMSE[];

    static void run(BufferedReader instances, BufferedReader weights) throws IOException {
        // read file, tokenize elements.
        controller(instances, weights);
    }

    public static void controller(BufferedReader instances, BufferedReader weights) throws IOException {
        // Parameter object holds all input fromt he user from a file.
        parameter par = createParameterObj(weights);
        RMSE = new double[par.numExamples];

        printInitialParameters(par);
        //  ExampleObj holds a whole instance of a neural network with all layers.
        createExampleObj(instances, par);
        //  Initial Weights are now stored into the Dynamic arrays.
        getWeights(par);
        for (int k = 1; k < 10001; k++)
        {
            //  loop n times here where n = number of epochs.
            for (int i = 0; i < par.numExamples; i++) {
                //  Initial weights are entered inside the objects.
                applyWeights(i);
                //  Backprop is being applied on each object within the list.
                applybackProp(par, i);
                //  Clear Weights so they can be updated
                clearWeights();
                //  Updated weights are entered into the objects.
                applyWeights(i);
                //printAll(par, i);
            }
            for (int i = 0; i < par.numExamples; i++)
            {
                //RMSE will be calculated here
                RMSE[i] = Math.sqrt((double) Math.pow((exampleList.get(i).targetOutput - exampleList.get(i).actualOutput), 2));
            }
            //  All objects are classified and results are printed
            classify(k, par.errorMargin);
        }
    }

    private static void printInitialParameters(parameter par) {
        System.out.println("=== Initial Parameters ===");
        System.out.println("Data File: 'activation.txt'");
        System.out.println("Number of Examples: " + par.numExamples);
        System.out.println();
        System.out.println("Number of Input Units:  " + par.inputUnits);
        System.out.println("Number of Hidden Units: " + par.hiddenUnits);
        System.out.println();
        System.out.println("Maximum Epochs: " + par.maxEpochs);
        System.out.println("Learning Rate:  " + par.learningRate);
        System.out.println("Error Margin:   " + par.errorMargin);
        System.out.println();
    }

    private static void createExampleObj(BufferedReader instances, parameter par) throws IOException {
        for (int countx = 0; countx < par.numExamples; countx++) {
            Scanner instSc = new Scanner(instances.readLine()).useDelimiter(",");
            exampleObj newExample = new exampleObj();

            inputNode threshHoldNode1 = new inputNode();
            hiddenNode threshHoldNode2 = new hiddenNode();
            threshHoldNode1.activation = -1;
            threshHoldNode2.activation = -1;
            newExample.inputNodes.add(threshHoldNode1);
            newExample.hiddenNodes.add(threshHoldNode2);

            for (int county1 = 0; county1 < par.inputUnits; county1++) {
                inputNode newNode = new inputNode();
                newNode.activation = Double.parseDouble(instSc.next());
                newExample.inputNodes.add(newNode);
            }
            newExample.targetOutput = Double.parseDouble(instSc.next());
            exampleList.add(newExample);

            for (int county2 = 0; county2 < par.hiddenUnits; county2++) {
                hiddenNode newNode = new hiddenNode();
                newExample.hiddenNodes.add(newNode);
            }
        }
    }

    private static void getWeights(parameter par) throws IOException {

        inputWeights = new double[par.inputUnits + 1][par.hiddenUnits];
        hiddenWeights = new double[par.hiddenUnits + 1];

        System.out.println("==== Initial Weights ====");
        System.out.println("Input (" + (par.inputUnits + 1) + ") --> " + "Hidden (" + (par.hiddenUnits + 1) + ")");
        for (int i = 0; i <= par.inputUnits; i++) {
            System.out.print(i);
            for (int j = 0; j < par.hiddenUnits; j++) {
                inputWeights[i][j] = (Math.random()*-5) + (Math.random()*5);//Double.parseDouble(weigSc.next());
                System.out.print(" " + inputWeights[i][j] + " ");
            }
            System.out.println();

        }
        System.out.println("Hidden (" + (par.hiddenUnits + 1) + ") --> " + "Output");
        for (int i = 0; i < par.hiddenUnits + 1; i++) {
            hiddenWeights[i] = (Math.random()*-5) + (Math.random()*5);

            System.out.println(i + " " + hiddenWeights[i]);
        }
        System.out.println();
    }

    private static void applyWeights(int i) {
        for (int j = 0; j < exampleList.get(i).inputNodes.size(); j++) {
            for (int k = 0; k < hiddenWeights.length - 1; k++) {
                exampleList.get(i).inputNodes.get(j).weights.add(inputWeights[j][k]);
            }
        }
        for (int j = 0; j < exampleList.get(i).hiddenNodes.size(); j++) {
            exampleList.get(i).hiddenNodes.get(j).weight = hiddenWeights[j];
        }
    }

    private static parameter createParameterObj(BufferedReader weights) throws IOException {
        String w = weights.readLine();
        Scanner weighSc = new Scanner(w).useDelimiter(",");

        int numEx = (Integer.parseInt(weighSc.next()));
        int inputUnits = (Integer.parseInt(weighSc.next()));
        int hiddenUnits = (Integer.parseInt(weighSc.next()));
        int maxEp = (Integer.parseInt(weighSc.next()));
        double leaRa = (Double.parseDouble(weighSc.next()));
        double errMa = (Double.parseDouble(weighSc.next()));

        parameter par = new parameter(numEx, inputUnits, hiddenUnits, maxEp, leaRa, errMa);
        return par;
    }

    private static void applybackProp(parameter par, int i) {
        //Propagate Forward
        propagateForward(i);
        //Propagate Backward
        propagateBackward(i);
        //Compute new Weights
        computeNewWeights(par, i);
    }

    private static void propagateForward(int i) {
        int index = 0;
        int numNodes = exampleList.get(index).hiddenNodes.size();

        for (int j = 1; j < numNodes; j++) {
            exampleList.get(i).hiddenNodes.get(j).activation = activate(exampleList.get(i).inputNodes, index);
            index++;
        }
        index = 0;

        exampleList.get(i).actualOutput = activate(exampleList.get(i).hiddenNodes, index);
    }

    private static void propagateBackward(int i) {

        double outputAct = exampleList.get(i).actualOutput;
        double targetOut = exampleList.get(i).targetOutput;
        exampleList.get(i).error = outputAct * (1 - outputAct) * (targetOut - outputAct);

        int index = 0;
        int numNodes = exampleList.get(index).hiddenNodes.size();

        for (int j = 1; j < numNodes; j++) {
            double hiddenAct = exampleList.get(i).hiddenNodes.get(j).activation;
            double hiddenWeight = exampleList.get(i).hiddenNodes.get(j).weight;
            exampleList.get(i).hiddenNodes.get(j).error = hiddenAct * (1 - hiddenAct) * hiddenWeight * exampleList.get(i).error;
        }
    }

    private static void computeNewWeights(parameter par, int k) {
        for (int i = 0; i < par.inputUnits + 1; i++) {
            for (int j = 0; j < par.hiddenUnits; j++) {
                double oldWeight = Double.parseDouble(exampleList.get(k).inputNodes.get(i).weights.get(j).toString());
                double outError = exampleList.get(k).hiddenNodes.get(j + 1).error;
                double activation = exampleList.get(k).inputNodes.get(i).activation;
                inputWeights[i][j] = oldWeight + par.learningRate * outError * activation;
            }
        }
        for (int i = 0; i < par.hiddenUnits + 1; i++) {
            double oldWeight = exampleList.get(k).hiddenNodes.get(i).weight;
            double outError = exampleList.get(k).error;
            double activation = exampleList.get(k).hiddenNodes.get(i).activation;
            hiddenWeights[i] = oldWeight + par.learningRate * outError * activation;
        }
    }

    private static double activate(LinkedList input, int index) {
        double denom = 0.0;
        double eq = 0.0;

        if (input.get(0) instanceof inputNode) {
            LinkedList<inputNode> inputL = input;
            for (int i = 0; i < inputL.size(); i++) {
                double weight = Double.parseDouble(inputL.get(i).weights.get(index).toString());
                double curAct = (inputL.get(i).activation);
                denom += weight * curAct;
            }
            eq = 1 / (1 + Math.exp(-denom));
        }

        if (input.get(0) instanceof hiddenNode) {
            LinkedList<hiddenNode> hiddenL = input;
            for (int i = 0; i < hiddenL.size(); i++) {
                double weight = hiddenL.get(i).weight;
                double curAct = hiddenL.get(i).activation;
                denom += weight * curAct;
            }
            eq = 1 / (1 + Math.exp(-denom));
        }
        return eq;
    }

    private static void printAll(parameter par, int i) {
        System.out.println("\n|------------------------------|");
        System.out.println("|Network Activation and Weights|");
        System.out.println("|------------------------------|");

        System.out.println("\nOutput Node");
        System.out.println("-----------");
        System.out.println("Actual Output: " + exampleList.get(i).actualOutput);
        System.out.println("Target Output: " + exampleList.get(i).targetOutput);
        System.out.println("Current Error: " + exampleList.get(i).error);
        System.out.println("------------------------");
        System.out.println("\nHidden Nodes");
        System.out.println("------------");
        for (int j = 0; j < par.hiddenUnits + 1; j++) {
            System.out.println("Activation: " + exampleList.get(i).hiddenNodes.get(j).activation);
            System.out.println("Weight: " + exampleList.get(i).hiddenNodes.get(j).weight);

        }
        System.out.println("\nInput Nodes");
        System.out.println("------------");
        for (int j = 0; j < par.inputUnits + 1; j++) {
            System.out.println("Activation: " + exampleList.get(i).inputNodes.get(j).activation);
            System.out.println("Weights: " + exampleList.get(i).inputNodes.get(j).weights.toString());
        }
        System.out.println("|------------------------------|");

    }

    private static void clearWeights() {
        for (int i = 0; i < exampleList.size(); i++) {
            for (int j = 0; j < exampleList.get(i).inputNodes.size(); j++) {
                exampleList.get(i).inputNodes.get(j).weights.clear();
            }
            for (int j = 0; j < exampleList.get(i).hiddenNodes.size(); j++) {
                exampleList.get(i).hiddenNodes.get(j).weight = 0.0;
            }

        }
    }

    private static void classify(int epochNum, double errorMargin) {
        int numClassified = 0;
        double maxRMSE = 0.0;
        double avgRMSE = 0.0;
        double percentCorrect = 0.0;

        for (int i = 0; i < RMSE.length; i++) {

            if (RMSE[i] > maxRMSE) {
                maxRMSE = RMSE[i];
            }
            if (RMSE[i] < errorMargin) {
                numClassified++;
            }
            avgRMSE += RMSE[i];
        }
        Double finalavgRMSE = (avgRMSE / (double) RMSE.length);
        percentCorrect = (double) numClassified / (double) RMSE.length;

        System.out.println("***** Epoch " + epochNum + " *****");
        System.out.println("Maximum RMSE: " + maxRMSE);
        System.out.println("Average RMSE: " + finalavgRMSE);
        System.out.println("Percent Correct: " + percentCorrect * 100 + "%");
    }
}
