%%
%Unsupervised feature learning on 3 author dataset

load('xtrain_glove.mat')
load('xvalid_glove.mat')

load('ytrain.mat')
load('yvalid.mat')
ytrain = double(ytrain');
yvalid = double(yvalid');
%Concatanete training and testing data
[a, b] = size(xtrain_glove);
[c, d] = size(xvalid_glove);
all_data = [xtrain_glove; xvalid_glove];
%Take the normr of the data
data = normr(all_data);
%We need to do K-means Clustering for K=1000
numClusters = 2000;
[centers, assignments] = vl_kmeans(data', numClusters,'Initialization', 'plusplus','Algorithm', 'ANN');

%Compare the distance for 1000 center points
centers = normc(centers);
train_features = zeros(a,numClusters);
for i = 1:a
    data_train = data(i,:);
    
    for l = 1:size(data_train,1)
        dist = zeros(1,numClusters);
        for j=1:numClusters
            dist(j) = dist_cosine(data_train(l,:),centers(:,j)');
        end
        mean_dist = mean(dist);
        zero_index = find(dist > mean_dist);
        dist(zero_index) = 0;
        train_features(i,:) = train_features(i,:) + dist;
    end
    train_features(i,:) = train_features(i,:) ./size(data_train,1);
    
    i
    
end

%Same steps for the testing data 
test_features = zeros(c,numClusters);
for i = 1:c
    data_test = data(i+a,:);
    
    for l = 1:size(data_test,1)
        dist = zeros(1,numClusters);
        for j=1:numClusters
            dist(j) = dist_cosine(data_test(l,:),centers(:,j)');
        end
        mean_dist = mean(dist);
        zero_index = find(dist > mean_dist);
        dist(zero_index) = 0;
        test_features(i,:) = test_features(i,:) + dist;
    end
    test_features(i,:) = test_features(i,:) ./size(data_test,1);
    i
end

%Murat hoca normalization technique
bow_mfw = train_features;
bowtst = test_features;
[n, d] = size(train_features);
[n2, d2] = size(test_features);

bowtst=bowtst./(sum(bowtst,2)*ones(1,size(bowtst,2)));
bow_mfw=bow_mfw./(sum(bow_mfw,2)*ones(1,size(bow_mfw,2)));
bow=[bow_mfw;bowtst];
for i=1:size(bow,2)
    bow(:,i)=(bow(:,i)-min(bow(:,i)))/max(bow(:,i)-min(bow(:,i)));
end
bow_train = bow(1:n,:);
bow_test = bow((n+1):end, :);

%Build a Model
model = train(ytrain, sparse(bow_train),['-s 1','-C 5']);
[predicted_label] = predict(ones(size(bow_test,1),1), sparse(bow_test), model);

count=0;
for i=1:c
    if predicted_label(i,1)==yvalid(i)
        count = count +1;
    end
end
Accuracy = count/c*100

result=confusionmatStats(yvalid, predicted_label(:,1));
mean(result.Fscore)