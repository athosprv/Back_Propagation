/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */



/**
 *
 * @author Ath
 */
public class parameter {

        int numExamples, maxEpochs, inputUnits, hiddenUnits;
        double learningRate, errorMargin;

        parameter(int numEx, int inputUnits, int hiddenUnits, int maxEp, double leaRa, double errMa)
        {
            numExamples = numEx;
            this.inputUnits = inputUnits;
            this.hiddenUnits = hiddenUnits;
            maxEpochs = maxEp;
            learningRate = leaRa;
            errorMargin = errMa;
        }

    }
