package model;

import methods.*;
import networks.*;

/**
 * Created by ledenev.p on 20.08.2015.
 */
public class TrainingBuilderFactory {

    // no multithreads support
    public static TrainingBuilder createRBFTrainer() {
        NetworkBuilder networkBuilder = new RBFNetworkBuilder();
        return new SVDTrainingBuilder(networkBuilder);
    }

    public static TrainingBuilder createResilientPropagationTrainer() {
        NetworkBuilder networkBuilder = new PerceptronBuilder();
        return new ResilientPropagationTrainingBuilder(networkBuilder);
    }

    // need to much memory
    public static TrainingBuilder createLevenbergMarquardtTrainer() {
        NetworkBuilder networkBuilder = new PerceptronBuilder();
        return new LevenbergMarquardtTrainingBuilder(networkBuilder);
    }

    // no multithreads support
    public static TrainingBuilder createNelderMeadTrainer() {
        NetworkBuilder networkBuilder = new PerceptronBuilder();
        return new NelderMeadTrainingBuilder(networkBuilder);
    }

    // no multithreads support
    public static TrainingBuilder createSVNSearchTrainer() {
        NetworkBuilder networkBuilder = new SVMBuilder();
        return new SVMSearchTrainingBuilder(networkBuilder);
    }

    // very slow convergence
    public static TrainingBuilder createScaledConjugateGradientTrainer() {
        NetworkBuilder networkBuilder = new PerceptronBuilder();
        return new ScaledConjugateGradientTrainingBuilder(networkBuilder);
    }

    // need some more details in input data
    public static TrainingBuilder createPNNTrainer() {
        NetworkBuilder networkBuilder = new PNNBuilder();
        return new PNNTrainingBuilder(networkBuilder);
    }

    // need more input data preparation
    public static TrainingBuilder createBayesianTrainer() {
        NetworkBuilder networkBuilder = new BayesianNetworkBuilder();
        return new BayesianTrainingBuilder(networkBuilder);
    }
}
