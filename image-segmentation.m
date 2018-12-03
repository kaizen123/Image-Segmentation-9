img=imread('face_2d.jpg')
%convert the color values in img to rgb2lab color space
form = makecform('srgb2lab');
lab_he = applycform(img,form);

ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);

nColors = 3;
% repeat the clustering 3 times to avoid local minima
[cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
                                      'Replicates',3);
pixel_labels = reshape(cluster_idx,nrows,ncols);
figure(1),subplot(1,4,1),imshow(pixel_labels,[]), title('image labeled by cluster index');

segmented_images = cell(1,3);
rgb_label = repmat(pixel_labels,[1 1 3]);

for k = 1:nColors
    color = img;
    color(rgb_label ~= k) = 0;
    segmented_images{k} = color;
end

figure(1),subplot(1,4,2),imshow(segmented_images{1}), title('objects in cluster 1');
figure(1),subplot(1,4,3),imshow(segmented_images{2}), title('objects in cluster 2');
figure(1),subplot(1,4,4),imshow(segmented_images{3}), title('objects in cluster 3');
