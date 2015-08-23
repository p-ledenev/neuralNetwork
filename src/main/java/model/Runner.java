package model;

import methods.*;
import org.encog.*;
import org.encog.ml.data.*;
import org.encog.ml.data.specific.*;
import org.encog.ml.train.*;
import org.encog.util.csv.*;

/**
 * Created by DiKey on 14.08.2015.
 */
public class Runner {

    public static String dataPath = "f:\\Teddy\\Alfa\\java\\v1.0\\neuralNetwork\\data\\";
    //public static String dataPath = "d:\\Projects\\Alfa\\java\\nn\\neuralNetwork\\data\\";
    //public static String dataPath = "./";

    public static void main(String[] args) throws Throwable {

        String fileName = "";
        //fileName = "usd_50inputs_max-min_2010.txt";
        fileName = "usd_50inputs_app_2010.txt";

        Log.info("Training data: " + fileName);
        MLDataSet trainingSet = new CSVNeuralDataSet(dataPath + fileName, 50, 2, false, new CSVFormat('.', ';'), false);

        TrainingBuilder trainingBuilder;
        //trainingBuilder = TrainingBuilderFactory.createRBFTrainer();
        trainingBuilder = TrainingBuilderFactory.createResilientPropagationTrainer();
        //trainingBuilder = TrainingBuilderFactory.createNelderMeadTrainer();
        //trainingBuilder = TrainingBuilderFactory.createLevenbergMarquardtTrainer();
        //trainingBuilder = TrainingBuilderFactory.createPNNTrainer();
        //trainingBuilder = TrainingBuilderFactory.createScaledConjugateGradientTrainer();
        //trainingBuilder = TrainingBuilderFactory.createBayesianTrainer();
        //trainingBuilder = TrainingBuilderFactory.createSVNSearchTrainer();

        BasicTraining training = trainingBuilder.build(trainingSet);
        training.setError(1.);

        Log.info("Starting training");
        for (int i = 1; training.getError() > 0.01; i++) {
            training.iteration();
            Log.info("Iteration: " + i + "; error: " + training.getError());

            if (i % 200 == 0) {
                Log.info("Saving network");
                training.finishTraining();
                trainingBuilder.saveNetwork(fileName);
            }
        }

        Encog.getInstance().shutdown();
    }


}
