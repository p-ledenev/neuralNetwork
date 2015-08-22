package methods;

import model.*;
import networks.*;
import org.encog.ml.*;
import org.encog.ml.data.*;
import org.encog.ml.train.*;

/**
 * Created by ledenev.p on 18.08.2015.
 */
public abstract class TrainingBuilder {

    private BasicTraining training;

    private NetworkBuilder networkBuilder;

    public TrainingBuilder(NetworkBuilder networkBuilder) {
        this.networkBuilder = networkBuilder;
    }

    public BasicTraining build(MLDataSet trainingSet) {

        training = createTraining(networkBuilder.build(trainingSet), trainingSet);

        Log.info("Training with " + training.getClass().getSimpleName());

        return training;
    }

    public void saveNetwork() {
        networkBuilder.saveNetwork();
    }

    protected abstract BasicTraining createTraining(BasicML network, MLDataSet trainingSet);
}
