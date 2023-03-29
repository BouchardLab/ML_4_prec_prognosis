############## This is a R script for subset selection of both features and patients
############## The results of this script are included in the "data/" subfolder:
############## 
############## 
############## 
############## 
############## 

cleanup = function(mat) {
    xx = mat
    yy = matrix(NA,nrow=dim(xx)[1],ncol=0);
    for (ii in 1:dim(xx)[2]){
        dd = xx[,ii];
        if (is.numeric(dd)){yy=cbind(yy,dd);colnames(yy)[dim(yy)[2]]=colnames(xx)[ii];}    
        if (is.factor(dd)){
            if (is.ordered(dd)){
                zz =as.numeric(dd);yy=cbind(yy,zz);colnames(yy)[dim(yy)[2]]=colnames(xx)[ii];
            } else {
                nlevels = length(levels(dd));
                if (nlevels<3){
                    zz =as.numeric(dd);yy=cbind(yy,zz);colnames(yy)[dim(yy)[2]]=colnames(xx)[ii];
                } else {
                    cat("nlevels is", nlevels, "for", colnames(xx)[ii], "\n")
                    for (jj in 1:nlevels){
                        zz = matrix(0,nrow=dim(xx)[1],ncol=1);
                        zz[dd==levels(dd)[jj]] = 1;
                        yy = data.frame(yy,zz);
                        colnames(yy)[dim(yy)[2]] = paste(colnames(xx)[ii],'_',levels(dd)[jj],sep='');
                    }
                }
            }
        }
    }
    rownames(yy) = rownames(xx);
    ss = apply(yy,2,sd);
    l = length(which(ss==0)); 
    if(l > 0){
        cat("found", l, "invariant columns", '\n');
        #cat(colnames(yy)[ss==0], '\n');
    }
    yy = yy[,ss>0];
    return(yy)
}

is.invariant = function(col){
    na=is.na(col); 
    ss = sd(col[!na]); 
    return(ss == 0);
}


########## read preprocessed datamatrix in the "data/" subfolder
pre_data_matrix = read.csv(file = "data/preprocessed_data_matrix.csv",header=TRUE);
rownames(pre_data_matrix) = pre_data_matrix[,1];
pre_data_matrix = pre_data_matrix[,2:dim(pre_data_matrix)[2]];

mask_feature_outcome = read.table('data/mask_feature_outcome.txt',sep='\t')
########## data occupancy matrix
x = is.na(pre_data_matrix);
x[x==TRUE] = 1;
x[x==FALSE] = 0;

# rows that have one or more NA
inc_rows = apply(pre_data_matrix, 1, is.na)

########## dendrogram of occupancy matrix
hm = heatmap(x,scale='none');
x = x[hm$rowInd,hm$colInd];

########## manually remove patients and features from the occupancy matrix
rowrmv=0*(1:dim(x)[1]);
rowrmv[c(1:25,37,51,67:68,69,143,146,150,153,170:172,178,188,205:213,216:229,254:299,318,320,325,337,360:dim(x)[1])] = 1;
colrmv=0*(1:dim(x)[2]);
colrmv[c(1:62,72:75,77:82,90:92,236:237,241,243,346:384,386:392,398:403,405:406,425:427,429:430,433:434,439,442:444,479,469,513:516,521:526,529,531:532,534,605:dim(x)[2])] = 1;

########## generate files of data matrix, patient_id, and feature_name for outcome features ##############################
########## a categorical feature will be expanded to N binary features (N is the number of categories) -- see cleanup function

write.table(hm$rowInd[rowrmv<1] - 1, file='data/keep_rows.txt', row.names=FALSE, col.names=FALSE)
write.table(hm$colInd[colrmv<1] - 1, file='data/keep_cols.txt', row.names=FALSE, col.names=FALSE)

# remove invariant rows from "good" rows

cat(length(which(colrmv == 1)), "columns to remove before\n");

prelim_rows = pre_data_matrix[sort(hm$rowInd[rowrmv<1]),];
invariant_cols = which(apply(prelim_rows, 2, is.invariant) == TRUE);
colrmv[invariant_cols] = 1
cat(length(which(colrmv == 1)), "columns to remove after\n");

data_matrix = pre_data_matrix[sort(hm$rowInd[rowrmv<1]),sort(hm$colInd[colrmv<1])];


xy_mask = mask_feature_outcome[sort(hm$colInd[colrmv<1]),1];


data_matrix_outcome = data_matrix[,xy_mask==1];
data_matrix_biomarker = data_matrix[,xy_mask==0];

yy = cleanup(data_matrix_outcome)
#xx = data_matrix_outcome;
#yy = matrix(NA,nrow=dim(xx)[1],ncol=0);
#for (ii in 1:dim(xx)[2]){
#    dd = xx[,ii];
#    if (is.numeric(dd)){yy=cbind(yy,dd);colnames(yy)[dim(yy)[2]]=colnames(xx)[ii];}    
#    if (is.factor(dd)){
#        if (is.ordered(dd)){
#            zz =as.numeric(dd);yy=cbind(yy,zz);colnames(yy)[dim(yy)[2]]=colnames(xx)[ii];
#        } else {
#            nlevels = length(levels(dd));
#            if (nlevels<3){
#                zz =as.numeric(dd);yy=cbind(yy,zz);colnames(yy)[dim(yy)[2]]=colnames(xx)[ii];
#            } else {
#                for (jj in 1:nlevels){
#                    zz = matrix(0,nrow=dim(xx)[1],ncol=1);
#                    zz[dd==levels(dd)[jj]] = 1;
#                    yy = data.frame(yy,zz);
#                    colnames(yy)[dim(yy)[2]] = paste(colnames(xx)[ii],'_',levels(dd)[jj],sep='');
#                }
#            }
#        }
#    }
#}
#rownames(yy) = rownames(xx);
#ss = apply(yy,2,sd);
#yy = yy[,ss>0];
write.table(yy,file='data/data_matrix_subset_outcome.txt',sep='\t',row.names=FALSE,col.names=FALSE);
write.table(rownames(yy),file='data/patient_id.txt',sep='\t',row.names=FALSE,col.names=FALSE,quote=FALSE);
write.table(colnames(yy),file='data/feature_name_subset_outcome.txt',sep='\t',row.names=FALSE,col.names=FALSE,quote=FALSE);


########## generate files of data matrix, patient_id, and feature_name for biomarker features #####################
########## a categorical feature will be expanded to N binary features (N is the number of categories) 

yy = cleanup(data_matrix_biomarker)
#xx = data_matrix_biomarker;
#yy = matrix(NA,nrow=dim(xx)[1],ncol=0);
#for (ii in 1:dim(xx)[2]){
#    dd = xx[,ii];
#    if (is.numeric(dd)){yy=cbind(yy,dd);colnames(yy)[dim(yy)[2]]=colnames(xx)[ii];}    
#    if (is.factor(dd)){
#        if (is.ordered(dd)){
#            zz =as.numeric(dd);yy=cbind(yy,zz);colnames(yy)[dim(yy)[2]]=colnames(xx)[ii];
#        } else {
#            nlevels = length(levels(dd));
#            if (nlevels<3){
#                zz =as.numeric(dd);yy=cbind(yy,zz);colnames(yy)[dim(yy)[2]]=colnames(xx)[ii];
#            } else {
#                for (jj in 1:nlevels){
#                    zz = matrix(0,nrow=dim(xx)[1],ncol=1);
#                    zz[dd==levels(dd)[jj]] = 1;
#                    yy = data.frame(yy,zz);
#                    colnames(yy)[dim(yy)[2]] = paste(colnames(xx)[ii],'_',levels(dd)[jj],sep='');
#                }
#            }
#        }
#    }
#}
#rownames(yy) = rownames(xx);
#ss = apply(yy,2,sd);
#yy = yy[,ss>0];
write.table(yy,file='data/data_matrix_subset_biomarker.txt',sep='\t',row.names=FALSE,col.names=FALSE);
write.table(colnames(yy),file='data/feature_name_subset_biomarker.txt',sep='\t',row.names=FALSE,col.names=FALSE,quote=FALSE);

########## save the discarded samples
# the data points we are discarding
#print(dim(pre_data_matrix))
#print(max(sort(hm$rowInd[rowrmv==1])))
#data_matrix_disc = pre_data_matrix[sort(hm$rowInd[rowrmv==1]),sort(hm$colInd[colrmv<1])];
#data_matrix_outcome_disc = data_matrix_disc[,xy_mask==1];
#data_matrix_biomarker_disc = data_matrix_disc[,xy_mask==0];
#
#yy = cleanup(data_matrix_outcome_disc)
#write.table(yy,file='data/data_matrix_discarded_outcome.txt',sep='\t',row.names=FALSE,col.names=FALSE);
#write.table(rownames(yy),file='data/patient_id_discarded.txt',sep='\t',row.names=FALSE,col.names=FALSE,quote=FALSE);
##
#yy = cleanup(data_matrix_biomarker_disc)
#write.table(yy,file='data/data_matrix_discarded_biomarker.txt',sep='\t',row.names=FALSE,col.names=FALSE);
