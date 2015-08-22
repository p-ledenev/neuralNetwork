package methods;

import networks.*;
import org.encog.ml.*;
import org.encog.ml.bayesian.*;
import org.encog.ml.bayesian.training.*;
import org.encog.ml.data.*;
import org.encog.ml.train.*;

/**
 * Created by ledenev.p on 20.08.2015.
 */
public class BayesianTrainingBuilder extends TrainingBuilder {

    public BayesianTrainingBuilder(NetworkBuilder networkBuilder) {
        super(networkBuilder);
    }

    @Override
    protected BasicTraining createTraining(BasicML network, MLDataSet trainingSet) {
        return new TrainBayesian((BayesianNetwork) network, trainingSet, 10);
    }
}
