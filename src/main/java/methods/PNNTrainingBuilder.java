package methods;

import networks.*;
import org.encog.ml.*;
import org.encog.ml.data.*;
import org.encog.ml.train.*;
import org.encog.neural.networks.training.pnn.*;
import org.encog.neural.pnn.*;

/**
 * Created by ledenev.p on 20.08.2015.
 */
public class PNNTrainingBuilder extends TrainingBuilder {

    public PNNTrainingBuilder(NetworkBuilder networkBuilder) {
        super(networkBuilder);
    }

    @Override
    protected BasicTraining createTraining(BasicML network, MLDataSet trainingSet) {
        return new TrainBasicPNN((BasicPNN) network, trainingSet);
    }
}
