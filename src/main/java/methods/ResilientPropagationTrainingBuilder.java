package methods;

import networks.*;
import org.encog.ml.*;
import org.encog.ml.data.*;
import org.encog.ml.train.*;
import org.encog.neural.networks.*;
import org.encog.neural.networks.training.propagation.resilient.*;

/**
 * Created by ledenev.p on 18.08.2015.
 */
public class ResilientPropagationTrainingBuilder extends TrainingBuilder {

    public ResilientPropagationTrainingBuilder(NetworkBuilder networkBuilder) {
        super(networkBuilder);
    }

    @Override
    protected BasicTraining createTraining(BasicML network, MLDataSet trainingSet) {

        ResilientPropagation training = new ResilientPropagation((BasicNetwork) network, trainingSet);
        training.setThreadCount(2);

        return training;
    }
}
