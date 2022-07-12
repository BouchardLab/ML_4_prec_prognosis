############## This is a R script for curation of the original TRACKTBI_Pilot_DEID_02.22.18v2.csv
############## The results of this script is preprocessed datamatrix and the mask for the outcome features in the "data/" subfolder


#########read the original TBI csv file 
ann = read.csv('TRACKTBI_Pilot_DEID_02.22.18v2.csv');
npatient = dim(ann)[1];
nfeature = dim(ann)[2];
x = ann;

############# the followings are 4 functions to be used in the main script #####################################
factornas <- function(z) {######### find all NA valuess in categorical features and replace them with NA
    z[z==" "] = NA;
    z[z=="?"] = NA;
    z[z=="N/A"] = NA;
    z[z=="N/A (lives alone)"] = NA;
    z[z=="Unknown"] = NA;
    z[z=="Don't Know"] = NA;
    z[z=="Not Applicable"] = NA;
    z[z=="Unk"] = NA;
    z[z=="ED Discharge pupils not done"] = NA;
    z = factor(z);
}

numericnas <- function(z) {######### find all NA valuess in numeric features and replace them with NA
    z[z==" "] = NA;
    z[z=="Unknown"] = NA;
    z[z=="Not Applicable"] = NA;
    z[z=="Unk"] = NA;
    z[z=="ED Discharge pupils not done"] = NA;
    z = as.numeric(as.matrix(z));
}

factorgiven <- function(z,npatient) {######### find all NA valuess in "given" features and replace them with NA
    d=0*(1:npatient)-1;
    d[z==" "] = NA;
    d[z=="Unknown"] = NA;
    d[z=="Not Given"] = 0;
    d[z=="Given"] = 1;
    return(d)
}

numtime <- function(z,npatient) {########## reformat all time strings to numeric values in minutes
    d=0*(1:npatient)-1;
    z = as.matrix(z);
    for (kk in 1:npatient){#cat(z[kk]);cat('\n');cat(kk);cat('\n')
        ss = z[kk];
        if(z[kk]==" ") d[kk]=NA;
        if(z[kk]=="Oops! time < 0") d[kk]=NA;
        if(length(grep("days",z[kk]))>0){
            ss = strsplit(z[kk],' days ')[[1]];
            d[kk] = as.numeric(ss[1])*24*60;
            if (ss[1]=="I") d[kk] = 1*24*60;
            ss = ss[2];
            if(length(grep(":",ss))>0){
                ss = strsplit(ss,':')[[1]];
                d[kk] = d[kk] + as.numeric(ss[1])*60 + as.numeric(ss[2]);if(d[kk]<0) d[kk]=NA;
            }
        }
        if(length(grep(" day ",z[kk]))>0){
            ss = strsplit(z[kk],' day ')[[1]];
            d[kk] = as.numeric(ss[1])*24*60;
            if (ss[1]=="I") d[kk] = 1*24*60;
            ss = ss[2];
            if(length(grep(":",ss))>0){
                ss = strsplit(ss,':')[[1]];
                d[kk] = d[kk] + as.numeric(ss[1])*60 + as.numeric(ss[2]);if(d[kk]<0) d[kk]=NA;
            }
        }
        if(length(grep("day",z[kk]))==0 & length(grep(":",z[kk]))>0){
            ss = strsplit(z[kk],':')[[1]];
            d[kk] = as.numeric(ss[1])*60 + as.numeric(ss[2]);if(d[kk]<0) d[kk]=NA;
        }
        
    }
    return(d)
}
################ end of functions #######################################

#################### manually reformat all column features when necessary################################

ii = 12; x[,ii] = factornas(ann[,ii]);
ii = 14; x[,ii] = factornas(ann[,ii]);
for (ii in 16:23) x[,ii] = factornas(ann[,ii]);
for (ii in 26:27) x[,ii] = factornas(ann[,ii]);
for (ii in 113:122) x[,ii] = factornas(ann[,ii]);
for (ii in 124:131) x[,ii] = factornas(ann[,ii]);
ii = 132; x[,ii] = numericnas(ann[,ii]);
for (ii in 133:134) x[,ii] = factornas(ann[,ii]);
ii = 135; x[,ii] = numtime(ann[,ii],npatient);
for (ii in 137:142) x[,ii] = factornas(ann[,ii]);
ii = 143; x[,ii] = numtime(ann[,ii],npatient);
ii = 144; x[,ii] = numtime(ann[,ii],npatient);
ii = 145; x[,ii] = factornas(ann[,ii]);
ii = 146; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="Unknown",ii]=NA;
          x[d=="None",ii]=0;x[d=="<1 minute",ii]=1;x[d=="1-29 minutes",ii]=2;x[d=="30-59 minutes",ii]=3;x[d=="1-24 hours",ii]=5;x[d==">24 hours",ii]=5;x[d==">7 days",ii]=6;
          x[,ii] = ordered(x[,ii]);
ii = 147; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="Unknown",ii]=NA;
          x[d=="None",ii]=0;x[d=="<0.5 Hours",ii]=1;x[d=="0.5-24 Hours",ii]=2;x[d==">24 Hours",ii]=3;
          x[,ii] = ordered(x[,ii]);
ii = 148; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="Unknown",ii]=NA;
          x[d=="No",ii]=0;x[d=="Suspected",ii]=1;x[d=="Yes",ii]=2;
          x[,ii] = ordered(x[,ii]);
ii = 149; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="Unknown",ii]=NA;
          x[d=="None",ii]=0;x[d=="<1 minute",ii]=1;x[d=="1-29 minutes",ii]=2;x[d=="30-59 minutes",ii]=3;x[d=="1-24 hours",ii]=5;x[d==">24 hours",ii]=5;x[d==">7 days",ii]=6;
          x[,ii] = ordered(x[,ii]);
ii = 150; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="Unknown",ii]=NA;
          x[d=="None",ii]=0;x[d=="<0.5 Hours",ii]=1;x[d=="0.5-24 Hours",ii]=2;x[d==">24 Hours",ii]=3;
          x[,ii] = ordered(x[,ii]);
ii = 151; x[,ii] = factornas(ann[,ii]);
ii = 156; x[,ii] = factornas(ann[,ii]);
ii = 159; x[,ii] = numtime(ann[,ii],npatient);
ii = 160; x[,ii] = factornas(ann[,ii]);
ii = 161; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="Unknown",ii]=NA;x[d=="Untestable",ii]=0;x[,ii]=as.numeric(x[,ii]);
ii = 163; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="Unknown",ii]=NA;x[d=="Untestable",ii]=0;x[,ii]=as.numeric((x[,ii]));
ii = 165; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="Unknown",ii]=NA;x[d=="Untestable",ii]=0;x[,ii]=as.numeric((x[,ii]));
ii = 168; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="Unknown",ii]=NA;
          x[d=="No Untestable Components",ii]=0;x[d=="One or More Untestable Components",ii]=1;
          x[,ii] = ordered(x[,ii]);
ii = 170; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="Unknown",ii]=NA;
          x[d=="Mild (13-15)",ii]=0;x[d=="Moderate (9-12)",ii]=1;x[d=="Severe (3-8)",ii]=2;
          x[,ii] = ordered(x[,ii]);
ii = 171; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="Unknown",ii]=NA;
          x[d=="Mild (13-15)",ii]=0;x[d=="Moderate (9-12)",ii]=1;x[d=="Severe (3-8)",ii]=2;
          x[,ii] = ordered(x[,ii]);
ii = 172; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="Unknown",ii]=NA;
          x[d=="Both pupils reactive",ii]=0;x[d=="One non-reactive pupil",ii]=1;x[d=="Both pupils unreactive",ii]=2;
          x[,ii] = ordered(x[,ii]);
ii = 174; x[,ii] = factornas(ann[,ii]);
ii = 176; x[,ii] = factornas(ann[,ii]);
ii = 181; x[,ii] = factornas(ann[,ii]);
ii = 184; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="Unknown",ii]=NA;x[d=="Untestable",ii]=0;x[,ii]=as.numeric((x[,ii]));
ii = 185; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="Unknown",ii]=NA;x[d=="Untestable",ii]=0;x[,ii]=as.numeric((x[,ii]));
ii = 186; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="Unknown",ii]=NA;x[d=="Untestable",ii]=0;x[,ii]=as.numeric((x[,ii]));
ii = 187; x[,ii] = factornas(ann[,ii]);
ii = 189; x[,ii] = numtime(ann[,ii],npatient);
for (ii in 190:191) x[,ii] = factornas(ann[,ii]);
ii = 192; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="Unknown",ii]=NA;
          x[d=="One Pupil Unreactive",ii]=1;x[d=="Both Pupils Unreactive",ii]=2;x[d=="3",ii]=3;
          x[,ii] = ordered(x[,ii]);
for (ii in 193:201) x[,ii] = factornas(ann[,ii]);
ii = 222; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="Unknown",ii]=NA;
          x[d=="No ED Fluids Given",ii]=0;x[d=="ED Fluids Given",ii]=1;
for (ii in 223:228) x[,ii] = factorgiven(ann[,ii],npatient);
for (ii in 229:233) x[,ii] = factornas(ann[,ii]);
ii = 234; x[,ii] = numtime(ann[,ii],npatient);
for (ii in 235:237) x[,ii] = factornas(ann[,ii]);
ii = 238; x[,ii] = numtime(ann[,ii],npatient);
for (ii in 257:272) x[,ii] = factornas(ann[,ii]);
for (ii in 289) x[,ii] = factornas(ann[,ii]);
for (ii in 291) x[,ii] = factornas(ann[,ii]);
ii = 293; x[,ii] = numtime(ann[,ii],npatient);
ii = 294; x[,ii] = numtime(ann[,ii],npatient);
for (ii in 295) x[,ii] = factornas(ann[,ii]);
for (ii in 316) x[,ii] = factornas(ann[,ii]);
ii = 317; x[,ii] = numtime(ann[,ii],npatient);
for (ii in 318:319) x[,ii] = factornas(ann[,ii]);
ii = 320; x[,ii] = numtime(ann[,ii],npatient);
for (ii in 321) x[,ii] = factornas(ann[,ii]);
ii = 323; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="missing from rPACS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 324; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="?",ii]=NA;x[,ii]=as.numeric(unlist(x[,ii]));
ii = 325; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="?",ii]=NA;x[d=="missing from rPACS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 326; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="?",ii]=NA;x[d=="15+",ii]=15;x[d==">15",ii]=15;x[,ii]=as.numeric((x[,ii]));
for (ii in 327) x[,ii] = factornas(ann[,ii]);
ii = 328; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="missing from rPACS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 330; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="missing from rPACS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 331; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="15 mm",ii]=15;x[d=="8 mm",ii]=8;x[,ii]=as.numeric((x[,ii]));
ii = 332; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="Indeterm",ii]=NA;x[d=="MRI [-]",ii]=0;x[d=="MRI [+]",ii]=1;
for (ii in 333:345) x[,ii] = factornas(ann[,ii]);
ii = 346; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="Participate a bit less",ii]=1;x[d=="Participate much less",ii]=2;x[d=="Unable to participate",ii]=3;
for (ii in 347:348) x[,ii] = factornas(ann[,ii]);
ii = 349; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="Constant",ii]=1;x[d=="Frequent",ii]=2;x[d=="Occasional",ii]=3;
for (ii in 350:355) x[,ii] = factornas(ann[,ii]);
ii = 382; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="Unknown",ii]=NA;x[d=="N/A",ii]=NA;x[d=="No",ii]=0;x[d=="Partial",ii]=1;x[d=="Full",ii]=2;
ii = 383; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="Unknown",ii]=NA;x[d=="N/A",ii]=NA;x[d=="No",ii]=0;x[d=="Partial",ii]=1;x[d=="Full",ii]=2;
ii = 384; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="None",ii]=0;x[d=="Minor",ii]=1;x[d=="Moderate",ii]=2;x[d=="Severe",ii]=3;
ii = 385; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="None",ii]=0;x[d=="Outpatient Rehab",ii]=1;x[d=="Inpatient Rehab",ii]=2;
for (ii in 386:398) x[,ii] = factornas(ann[,ii]);
ii = 399; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="Participate a bit less",ii]=1;x[d=="Participate much less",ii]=2;x[d=="Unable to participate",ii]=3;
for (ii in 400:401) x[,ii] = factornas(ann[,ii]);
ii = 402; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="Constant",ii]=1;x[d=="Frequent",ii]=2;x[d=="Occasional",ii]=3;
for (ii in 403:408) x[,ii] = factornas(ann[,ii]);
ii = 410; x[,ii] = numtime(ann[,ii],npatient);
ii = 436; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="Unknown",ii]=NA;x[d=="N/A",ii]=NA;x[d=="No",ii]=0;x[d=="Partial",ii]=1;x[d=="Full",ii]=2;
ii = 437; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="None",ii]=0;x[d=="Minor",ii]=1;x[d=="Moderate",ii]=2;x[d=="Severe",ii]=3;
ii = 438; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="None",ii]=0;x[d=="Outpatient Rehab",ii]=1;x[d=="Inpatient Rehab",ii]=2;
ii = 443; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="Strongly Disagree",ii]=0;x[d=="Slightly Disagree",ii]=1;x[d=="Disagree",ii]=2;
          x[d=="Neither Agree nor Disagree",ii]=3;x[d=="Slightly Agree",ii]=4;x[d=="Agree",ii]=5;x[d=="Strongly Agree",ii]=6;
ii = 444; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="Strongly Disagree",ii]=0;x[d=="Slightly Disagree",ii]=1;x[d=="Disagree",ii]=2;
          x[d=="Neither Agree nor Disagree",ii]=3;x[d=="Slightly Agree",ii]=4;x[d=="Agree",ii]=5;x[d=="Strongly Agree",ii]=6;
ii = 445; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="Strongly Disagree",ii]=0;x[d=="Slightly Disagree",ii]=1;x[d=="Disagree",ii]=2;
          x[d=="Neither Agree nor Disagree",ii]=3;x[d=="Slightly Agree",ii]=4;x[d=="Agree",ii]=5;x[d=="Strongly Agree",ii]=6;
ii = 446; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="Strongly Disagree",ii]=0;x[d=="Slightly Disagree",ii]=1;x[d=="Disagree",ii]=2;
          x[d=="Neither Agree nor Disagree",ii]=3;x[d=="Slightly Agree",ii]=4;x[d=="Agree",ii]=5;x[d=="Strongly Agree",ii]=6;
ii = 447; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="Strongly Disagree",ii]=0;x[d=="Slightly Disagree",ii]=1;x[d=="Disagree",ii]=2;
          x[d=="Neither Agree nor Disagree",ii]=3;x[d=="Slightly Agree",ii]=4;x[d=="Agree",ii]=5;x[d=="Strongly Agree",ii]=6;
ii = 485; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="Someone else is always with me to observe or supervise.",ii]=0;
          x[d=="Someone else is always around, but they only check on me now and then.",ii]=1;
          x[d=="Sometimes I am left alone for an hour or two.",ii]=2;
          x[d=="Sometimes I am left alone for most of the day.",ii]=3;
          x[d=="I have been left alone all day and all night, but someone checks in on me.",ii]=4;
          x[d=="I am left alone without anyone checking in on me.",ii]=5;
ii = 486; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="I do not need help going anywhere.",ii]=0;
          x[d=="I go to places on my own as long as they are familiar.",ii]=1;
          x[d=="Someone is always with me to help with remembering, decision making, or judgment when I go anywhere.",ii]=2;
          x[d=="I am restricted from leaving, even with someone else.",ii]=3;
ii = 489; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="None",ii]=0;x[d=="2-Jan",ii]=1;x[d=="4-Mar",ii]=2;x[d=="5 or more",ii]=3;
for (ii in 497) x[,ii] = factornas(ann[,ii]);
ii = 501; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="None",ii]=0;x[d=="2-Jan",ii]=1;x[d=="5-Mar",ii]=2;x[d=="6 or more",ii]=3;
for (ii in 536) x[,ii] = factornas(ann[,ii]);
for (ii in 596:609) x[,ii] = factornas(ann[,ii]);
ii = 610; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="Participate a bit less",ii]=1;x[d=="Participate much less",ii]=2;x[d=="Unable to participate",ii]=3;
for (ii in 611:612) x[,ii] = factornas(ann[,ii]);
ii = 613; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="Constant",ii]=1;x[d=="Frequent",ii]=2;x[d=="Occasional",ii]=3;
for (ii in 614:619) x[,ii] = factornas(ann[,ii]);
ii = 621; x[,ii] = numtime(ann[,ii],npatient);
for (ii in 647) x[,ii] = factornas(ann[,ii]);
ii = 648; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="Unknown",ii]=NA;x[d=="N/A",ii]=NA;x[d=="No",ii]=0;x[d=="Partial",ii]=1;x[d=="Full",ii]=2;
ii = 649; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="None",ii]=0;x[d=="Minor",ii]=1;x[d=="Moderate",ii]=2;x[d=="Severe",ii]=3;
for (ii in 650) x[,ii] = factornas(ann[,ii]);
ii = 651; x[,ii]=0*(1:npatient)-1;d=ann[,ii];x[d==" ",ii]=NA;x[d=="No Rehab",ii]=0;x[d=="Outpatient Rehab",ii]=1;x[d=="General Inpatient Rehab",ii]=2;x[d=="TBI Rehab",ii]=3;
for (ii in 673) x[,ii] = factornas(ann[,ii]);
ii = 728; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 729; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 730; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 731; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<0.55",ii]=0.55;x[d=="<0.56",ii]=0.56;x[,ii]=as.numeric((x[,ii]));
ii = 732; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<1.9",ii]=1.9;x[d=="<4.6",ii]=4.6;
          x[d=="2.6",ii]=4.6;x[d=="3.1",ii]=4.6;x[d=="3.4",ii]=4.6;x[d=="3.6",ii]=4.6;x[d=="4",ii]=4.6;x[d=="4.09999",ii]=4.6;x[,ii]=as.numeric((x[,ii]));
ii = 733; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 734; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 735; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 736; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 737; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 738; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 739; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 741; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<0.039",ii]=0.039;x[,ii]=as.numeric((x[,ii]));
ii = 743; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<3.8",ii]=3.8;
          x[d=="<8.7",ii]=8.7;x[d=="4",ii]=8.7;x[d=="<4.9",ii]=8.7;x[d=="<6.6",ii]=8.7;x[,ii]=as.numeric((x[,ii]));
ii = 744; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<3.0",ii]=3.0;
          x[d=="<7.0",ii]=7.0;x[d=="4",ii]=7.0;x[d=="4.5",ii]=7.0;x[d=="5.09999",ii]=7.0;x[d=="5.6",ii]=7.0;x[d=="6.7",ii]=7.0;x[,ii]=as.numeric((x[,ii]));
ii = 745; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<0.56",ii]=0.56;
          x[d=="0.4",ii]=0.56;x[d=="0.47",ii]=0.56;x[d=="0.53",ii]=0.56;x[,ii]=as.numeric((x[,ii]));
ii = 746; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 747; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<0.016",ii]=0.016;x[d=="<0.091",ii]=0.091;
          x[d=="0.03",ii]=0.091;x[d=="0.06",ii]=0.091;x[d=="0.08",ii]=0.091;x[d=="2.30E-02",ii]=0.091;x[d=="3.90E-02",ii]=0.091;x[d=="4.70E-02",ii]=0.091;x[d=="6.20E-0",ii]=0.091;x[,ii]=as.numeric((x[,ii]));
ii = 748; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<85",ii]=85;x[d=="<145",ii]=145;
          x[d=="94",ii]=145;x[d=="113",ii]=145;x[d=="122",ii]=145;x[d=="132",ii]=145;x[,ii]=as.numeric((x[,ii]));
ii = 749; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<0.063",ii]=0.063;x[d=="<0.084",ii]=0.084;x[,ii]=as.numeric((x[,ii]));
ii = 750; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 751; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<2.8",ii]=2.8;x[d=="<4.3",ii]=4.3;
          x[d==">3040",ii]=3040;x[d==">4020",ii]=4020;x[,ii]=as.numeric((x[,ii]));
ii = 752; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 753; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<5.4",ii]=5.4;x[d=="<48",ii]=48;
          x[,ii]=as.numeric((x[,ii]));x[(!is.na(x[,ii]))&(x[,ii]>5.4)&(x[,ii]<48),ii]=48;
ii = 754; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<21",ii]=21;x[d=="<88",ii]=88;x[,ii]=as.numeric((x[,ii]));
ii = 755; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<0.14",ii]=0.14;x[d=="<0.15",ii]=0.15;x[,ii]=as.numeric((x[,ii]));
ii = 756; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<0.016",ii]=0.016;x[d=="<0.064",ii]=0.064;x[d=="6.30E-02",ii]=0.064;x[,ii]=as.numeric((x[,ii]));
ii = 757; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<1.1",ii]=1.1;x[d=="<2.3",ii]=2.3;x[d=="1.4",ii]=2.3;x[,ii]=as.numeric((x[,ii]));
ii = 758; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 759; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<6.1",ii]=6.1;x[d=="<18",ii]=18;
          x[d==">1985",ii]=1985;x[,ii]=as.numeric((x[,ii]));x[(!is.na(x[,ii]))&(x[,ii]>6.1)&(x[,ii]<18),ii]=18;
ii = 760; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 761; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<0.21",ii]=0.21;x[d=="<0.22",ii]=0.22;x[,ii]=as.numeric((x[,ii]));
ii = 762; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<4.1",ii]=4.1;x[,ii]=as.numeric((x[,ii]));
ii = 764; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<1.5",ii]=1.5;x[d=="<2.5",ii]=2.5;x[,ii]=as.numeric((x[,ii]));
          x[(!is.na(x[,ii]))&(x[,ii]>1.5)&(x[,ii]<2.5),ii]=2.5;
ii = 765; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<0.0013",ii]=0.0013;x[d=="<0.0020",ii]=0.0020;x[,ii]=as.numeric((x[,ii]));
ii = 766; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<2.8",ii]=2.8;x[d=="<6.6",ii]=6.6;x[d=="2.9",ii]=6.6;x[,ii]=as.numeric((x[,ii]));
ii = 767; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<69",ii]=69;x[d=="<95",ii]=95;x[d=="70",ii]=95;x[d=="93",ii]=95;x[,ii]=as.numeric((x[,ii]));
ii = 768; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<8.3",ii]=8.3;x[d=="<41",ii]=41;x[,ii]=as.numeric((x[,ii]));
ii = 769; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<0.0051",ii]=0.0051;x[d=="<0.016",ii]=0.016;x[,ii]=as.numeric((x[,ii]));
ii = 770; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<29",ii]=29;x[d=="<35",ii]=35;x[,ii]=as.numeric((x[,ii]));
ii = 771; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<4.7",ii]=4.7;x[d=="<13",ii]=13;x[,ii]=as.numeric((x[,ii]));
          x[(!is.na(x[,ii]))&(x[,ii]>4.7)&(x[,ii]<13),ii]=13;
ii = 772; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<4.5",ii]=4.5;x[d=="<11",ii]=11;x[,ii]=as.numeric((x[,ii]));
          x[(!is.na(x[,ii]))&(x[,ii]>4.5)&(x[,ii]<11),ii]=11;
ii = 773; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<8.8",ii]=8.8;x[d=="<32",ii]=32;x[,ii]=as.numeric((x[,ii]));
ii = 774; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<2.9",ii]=2.9;x[,ii]=as.numeric((x[,ii]));
ii = 775; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<6.8",ii]=6.8;x[d=="<6.9",ii]=6.9;x[,ii]=as.numeric((x[,ii]));
ii = 776; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<0.28",ii]=0.28;x[,ii]=as.numeric((x[,ii]));x[(!is.na(x[,ii]))&(x[,ii]<0.28),ii]=0.28;
ii = 777; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<49",ii]=49;x[d=="<50",ii]=50;x[,ii]=as.numeric((x[,ii]));
ii = 778; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<6.0",ii]=6.0;x[d=="<6.2",ii]=6.2;x[,ii]=as.numeric((x[,ii]));
ii = 779; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<0.39",ii]=0.39;x[d=="<0.44",ii]=0.44;x[,ii]=as.numeric((x[,ii]));
ii = 780; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 781; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<2.9",ii]=2.9;x[d=="<5.0",ii]=5.0;x[,ii]=as.numeric((x[,ii]));
          x[(!is.na(x[,ii]))&(x[,ii]>2.9)&(x[,ii]<5.0),ii]=5.0;
ii = 782; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 783; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<0.40",ii]=0.40;x[d=="<1.0",ii]=1.0;x[,ii]=as.numeric((x[,ii]));
          x[(!is.na(x[,ii]))&(x[,ii]>0.40)&(x[,ii]<1.0),ii]=1.0;
ii = 784; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 785; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 786; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<29",ii]=29;x[d=="<42",ii]=42;x[,ii]=as.numeric((x[,ii]));
          x[(!is.na(x[,ii]))&(x[,ii]>29)&(x[,ii]<42),ii]=42;
ii = 787; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 788; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 789; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<34",ii]=34;x[,ii]=as.numeric((x[,ii]));
ii = 790; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<56",ii]=56;x[,ii]=as.numeric((x[,ii]));
ii = 791; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<186",ii]=186;x[,ii]=as.numeric((x[,ii]));x[(!is.na(x[,ii]))&(x[,ii]<186),ii]=186;
ii = 792; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d==">2980",ii]=2980;x[,ii]=as.numeric((x[,ii]));
ii = 793; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 794; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 795; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<0.013",ii]=0.013;x[d=="<0.017",ii]=0.017;x[d==">3.1",ii]=3.1;
          x[,ii]=as.numeric((x[,ii]));x[(!is.na(x[,ii]))&(x[,ii]>3.1),ii]=3.1;
ii = 796; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 797; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 798; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<116",ii]=116;x[d=="<142",ii]=142;x[d=="134",ii]=142;x[,ii]=as.numeric((x[,ii]));
ii = 799; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 800; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 802; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 803; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 804; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 805; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 806; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<14",ii]=14;x[d=="<23",ii]=23;x[d=="18",ii]=23;x[d=="22",ii]=23;x[,ii]=as.numeric((x[,ii]));
ii = 807; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<9.7",ii]=9.7;x[d=="<45",ii]=45;x[d=="14",ii]=45;x[,ii]=as.numeric((x[,ii]));
ii = 808; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<1.0",ii]=1.0;x[,ii]=as.numeric((x[,ii]));
ii = 810; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 811; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 813; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[,ii]=as.numeric((x[,ii]));
ii = 814; d=ann[,ii];x[,ii]=as.matrix(d);x[d==" ",ii]=NA;x[d=="NR",ii]=NA;x[d=="QNS",ii]=NA;x[d=="<12",ii]=12;x[d=="<17",ii]=17;x[,ii]=as.numeric((x[,ii]));




############### mark outcome features (value=1) #########################
mask_feature_outcome = 0*(1:nfeature);
mask_feature_outcome[333:727] = 1;

############## mark excluded features #########################
mask_feature_include = 0*(1:nfeature) + 1;
mask_feature_include[c(1:2)] = 0;
mask_feature_include[c(137)] = 0;
mask_feature_include[c(491)] = 0;

############### remove excluded features #########################
rownames(x) = ann[,1];
colnames(x) = colnames(ann);
x = x[,mask_feature_include>0];
mask_feature_outcome = mask_feature_outcome[mask_feature_include>0];
write.table(mask_feature_outcome,file='data/mask_feature_outcome.txt',sep='\t',row.names=FALSE,col.names=FALSE,quote=FALSE);

########### final pre-processed data matrix #########################
pre_data_matrix = x;

write.csv(pre_data_matrix, file = "data/preprocessed_data_matrix.csv")











































